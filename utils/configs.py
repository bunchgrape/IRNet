# Copyright 2022 CircuitNet. All rights reserved.

import argparse
import os
import sys
from .tools import *

sys.path.append(os.getcwd())


class Parser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--save_path", default="results")
        self.parser.add_argument("--pretrained", default=None)
        self.parser.add_argument("--max_iters", type=int, default=20000)
        self.parser.add_argument("--arg_file", default=None)
        self.parser.add_argument("--cpu", action="store_true")
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--eval_freq", type=int, default=10)
        self.parser.add_argument("--save_freq", type=int, default=20)

        ## logging and saver
        self.parser.add_argument("--log_freq", type=int, default=100)
        self.parser.add_argument("--result_dir", type=str, default="results/train", help="log/model root directory")
        self.parser.add_argument("--exp_id", type=str, default="", help="experiment id")
        self.parser.add_argument("--log_dir", type=str, default="log", help="log directory")
        self.parser.add_argument("--log_name", type=str, default="train.log", help="log file name")
        self.parser.add_argument("--eval_dir", type=str, default="eval", help="visualization directory")

        ## model hyperparameters
        self.parser.add_argument("--dataroot", default="./train_data")
        self.parser.add_argument("--ann_file_train", default="./index/train_N28.csv")
        self.parser.add_argument("--ann_file_test", default="./index/test_N28.csv")
        self.parser.add_argument("--dataset_type", default="IRDropDataset")
        self.parser.add_argument("--batch_size", type=int, default=2)
        self.parser.add_argument("--proportion", type=float, default=0.7, help="Proportion of training data")
        self.parser.add_argument("--model_type", default="MAVI")
        self.parser.add_argument("--in_channels", type=int, default=1, help="Innput channels")
        self.parser.add_argument("--out_channels", type=int, default=1, help="Output channels")
        self.parser.add_argument("--temporal_dim", type=int, default=24, help="Temporal feature dimension")
        self.parser.add_argument("--lr", type=float, default=1e-3)
        self.parser.add_argument("--weight_decay", type=float, default=1e-2)

        self.parser.add_argument("--eval_metric", default=["NRMS", "SSIM", "NMAE", "F1", "MAE", "R2"])
        self.parser.add_argument("--val_metric", default=["LpLoss", "MAE", "F1"])
        self.parser.add_argument("--loss_type", default=["LpLoss", "MAE", "MSE", "BCE", "SSIM"])
        self.parser.add_argument("--log_loss", default=["LpLoss"])
        self.parser.add_argument("--loss_code", default=[1, 0, 0, 0, 1], type=int, nargs="+", help="0.lp | 1.l1 | 2.l2 | 3.bce | 4.ssim")
        
        self.args = self.parser.parse_args()
        self.args.exp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + self.args.exp_id
        self.args.exp_id = "{}_{}_{}".format(self.args.exp_id, self.args.model_type, self.args.dataset_type)
