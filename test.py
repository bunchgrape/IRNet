import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from models.build_model import build_model
from utils.configs import Parser
import sys, os
from utils import *


def test():
    argp = Parser()
    args = argp.args
    arg_dict = vars(args)
    args.exp_id = args.exp_id + "_test"

    if args.arg_file is not None:
        with open(args.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    logger = setup_logger(args, sys.argv)

    ##
    ## Initialize dataset
    logger.info("===> Loading datasets")
    arg_dict["ann_file"] = arg_dict["ann_file_test"]
    arg_dict["test_mode"] = True
    dataset = build_dataset(arg_dict)

    logger.info("test dataset size: %d" % (len(dataset)))

    ##
    ## Initialize model parameters
    logger.info("===> Building model")
    model = build_model(args)
    if not arg_dict["cpu"]:
        model = model.cuda()

    # Build metrics
    metrics = {k: build_metric(k) for k in arg_dict["eval_metric"]}
    avg_metrics = {k: 0 for k in arg_dict["eval_metric"]}

    runtime_start = time.time()
    for feature, label, _ in tqdm(dataset, total=len(dataset), desc="Evaluation", unit="sample", leave=False):
        if arg_dict["cpu"]:
            input, target = feature, label
        else:
            input, target = feature.cuda(), label.cuda()

        prediction = model(input).clamp(min=0.0)

        for metric, metric_func in metrics.items():
            avg_metrics[metric] += metric_func(target.detach().cpu(), prediction.detach().cpu())

    runtime_end = time.time()
    logger.info("===> Total runtime: {:.5f} seconds".format(runtime_end - runtime_start))
    logger.info("===> per sample runtime: {:.5f} seconds".format((runtime_end - runtime_start) / len(dataset)))

    for metric, avg_metric in avg_metrics.items():
        logger.info("===> Avg. {}: {:.6f}".format(metric, avg_metric / len(dataset)))


if __name__ == "__main__":
    test()
