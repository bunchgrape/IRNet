import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from .SwinTransformer import *
from collections import OrderedDict
from .init_weights import load_state_dict

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale**2 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale**2),
        )

        x = self.norm(x)

        x = x.permute(0, 3, 1, 2)
        return x

class IRNetDual(nn.Module):
    def __init__(self, img_size=256, in_channels=1, out_channels=4, temporal_dim=24, embed_dim=96, patch_size=4, depths=[2, 2, 2, 2], **kwargs):
        super(IRNetDual, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_dim = temporal_dim

        self.embed_dim = embed_dim
        patch_size_3d = (2, patch_size, patch_size)
        patch_size_2d = (patch_size, patch_size)

        # B, 1, 24, 256, 256
        # B, 24, 256, 256
        self.num_decoder_layers = len(depths) - 1

        self.swin_unet_3d = SwinTransformerSys3D(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            temporal_dim=temporal_dim - 4, # Temporal features
            patch_size=patch_size_3d,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=[4, 8, 8],
        )

        self.swin_unet_2d = SwinTransformerSys(
            img_size=self.img_size,
            in_channels=temporal_dim,
            out_channels=self.out_channels,
            patch_size=patch_size_2d,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=[3, 6, 12, 24],
            window_size=[16, 16],
        )

        """ Fusing Decoder """
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            emb_dim = self.embed_dim * 2 ** (self.num_decoder_layers - i)
            up_emb_dim = self.embed_dim * 2 ** (self.num_decoder_layers - 1 - i)
            self.ups.append(UpConv(emb_dim, up_emb_dim))
            self.convs.append(ConvBlock(emb_dim, up_emb_dim))

        self.final_Expand = FinalPatchExpand_X4(
            dim_scale=patch_size,
            dim=self.embed_dim,
        )

        self.output = nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_t = x[:, :, 4 : self.temporal_dim, :, :]
        x_s = x.squeeze(1)

        """ ===================== Encoder ===================== """
        x_t, x_t_down = self.swin_unet_3d(x_t)
        x_s, x_s_down = self.swin_unet_2d(x_s)

        x_downsample = []
        for x1, x2 in zip(x_t_down, x_s_down):
            x_downsample.append(x1.mean(dim=2) * x2)
        x = x_t.mean(dim=2) * x_s

        """ ===================== Decoder ===================== """
        for i in range(self.num_decoder_layers):
            d = self.ups[i](x)
            s = x_downsample[self.num_decoder_layers - 1 - i]
            d = torch.cat((s, d), dim=1)
            x = self.convs[i](d)

        x = self.final_Expand(x)

        out = self.output(x)

        return out.squeeze(1)

    def init_weights(self, pretrained=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location="cpu")["state_dict"]
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)

                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. ' f"But received {type(pretrained)}.")
