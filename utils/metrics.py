# Copyright 2022 CircuitNet. All rights reserved.

from functools import wraps
from inspect import getfullargspec

import os
import os.path as osp
import numpy as np
import torch

from sklearn.metrics import f1_score, r2_score
from skimage.metrics import normalized_root_mse
from pytorch_msssim import ssim as ssim_pytorch

import utils.metrics as metrics

__all__ = ["lploss", "nmae", "mae", "ssim", "nrms", "f1", "r2",]


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def lploss(x, y, p=2):
    num_examples = x.size()[0]
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), p, 1)
    return torch.mean(diff_norms / y_norms)

def mae(img1, img2):
    return torch.nn.L1Loss()(img1, img2)

def nmae(img1, img2):
    normalized_mae = mae(img1, img2) / (torch.max(img1) - torch.min(img1)) if (torch.max(img1) - torch.min(img1)) != 0 else 0.05
    return normalized_mae

def nrms(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()
    if (img1.max() - img1.min()) == 0:
        return 0.05
    nrmse_value = normalized_root_mse( img1.flatten(), img2.flatten(), normalization="min-max")
    return nrmse_value

def ssim(img1, img2):
    scalar = img1.max()
    img1 = img1 / scalar
    img2 = img2 / scalar
    ssim_val = ssim_pytorch(img1.unsqueeze(1), img2.unsqueeze(1), data_range=1, size_average=True) # return (N,)
    return ssim_val

def f1(img1, img2):     
    ## label, prediction
    num_pred = img1.shape[0]
    f1_score_metric = 0.0
    for i in range(num_pred):
        true = img1[i].cpu().detach()
        pred = img2[i].cpu().detach()
        threshold_pred = torch.nanquantile(pred, 0.9).item()
        threshold_true = torch.nanquantile(true, 0.9).item()
        pred_binary = (pred >= threshold_pred).to(torch.int32).ravel()
        true_binary = (true >= threshold_true).to(torch.int32).ravel()
        f1_score_metric += f1_score(pred_binary, true_binary).item()
    f1_score_metric /= num_pred
    return f1_score_metric

def r2(img1, img2):
    return r2_score(img1.flatten().numpy(), img2.flatten().numpy())


def build_metric(metric_name):
    return metrics.__dict__[metric_name.lower()]