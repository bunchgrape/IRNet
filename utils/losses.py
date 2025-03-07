# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, pi
from pytorch_msssim import SSIM

import utils.losses as losses

__all__ = ["lploss", "mae", "mse", "bce", "ssim"]


## normalized Lp Loss
class lploss(nn.Module):
    def __init__(self, args, p=2, size_average=True, reduction=True):
        super(lploss, self).__init__()

        self.batch_size = args.batch_size

        # Dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x.view(self.batch_size, -1), y.view(self.batch_size, -1))


class mae(nn.Module):
    def __init__(self, args):
        super(mae, self).__init__()
        self.crit = nn.L1Loss()

    def __call__(self, x, y):
        return self.crit(x, y)


class mse(nn.Module):
    def __init__(self, args):
        super(mse, self).__init__()
        self.crit = nn.MSELoss()

    def __call__(self, x, y):
        return self.crit(x, y)


class bce(nn.Module):
    def __init__(self, args):
        super(bce, self).__init__()
        self.crit = nn.BCELoss()

    def __call__(self, x, y):
        return self.crit(x, y)


class ssim(nn.Module):
    def __init__(self, args, data_range=1, size_average=True, channel=1):
        super(ssim, self).__init__()
        self.crit = SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def __call__(self, x, y):
        return 1 - self.crit(x.unsqueeze(1), y.unsqueeze(1))


def build_loss(loss_name, args):
    return losses.__dict__[loss_name.lower()](args=args)


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class CosineRestartLr(object):
    def __init__(self, base_lr, periods, restart_weights=[1], min_lr=None, min_lr_ratio=None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))]

        self.base_lr = base_lr

    def annealing_cos(self, start: float, end: float, factor: float, weight: float = 1.0) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f"Current iteration {iteration} exceeds " f"cumulative_periods {cumulative_periods}")

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault("initial_lr", group["lr"])
            self.base_lr = [group["initial_lr"] for group in optimizer.param_groups]  # type: ignore
