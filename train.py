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


def train():
    argp = Parser()
    args = argp.args
    arg_dict = vars(args)
    if args.arg_file is not None:
        with open(args.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    if not os.path.exists(arg_dict["save_path"]):
        os.makedirs(arg_dict["save_path"])
    with open(os.path.join(arg_dict["save_path"], "args.json"), "wt") as f:
        json.dump(arg_dict, f, indent=4)
    logger = setup_logger(args, sys.argv)

    ##
    ## Initialize dataset
    logger.info("===> Loading datasets")
    arg_dict["ann_file"] = arg_dict["ann_file_train"]
    arg_dict["test_mode"] = False
    train_dataset = build_dataset(arg_dict)
    # Initialize validation dataset
    arg_dict["ann_file"] = arg_dict["ann_file_test"]
    arg_dict["test_mode"] = True
    val_dataset = build_dataset(arg_dict)
    logger.info("dataset size train: %d | val: %d" % (len(train_dataset), len(val_dataset)))

    ##
    ## Initialize model parameters
    logger.info("===> Building model")
    model = build_model(args)
    if not arg_dict["cpu"]:
        model = model.cuda()

    ##
    ## Build Loss
    losses = {k: build_loss(k, args) for k in args.loss_type}
    loss_code = {args.loss_type[i]: args.loss_code[i] for i in range(len(args.loss_type))}
    val_metrics = {k: build_metric(k) for k in args.val_metric}

    ##
    ## Build Optimzer
    model.requires_grad_(True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    ##
    ## Build lr scheduler
    args.max_iters = len(train_dataset) * args.num_epochs
    cosine_lr = CosineRestartLr(args.lr, [args.max_iters], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    ## =====================================================================
    ## Training
    iter_num = 0
    train_loss_curve = []
    for epoch_num in range(args.num_epochs):
        model.train()
        train_losses = {k: 0.0 for k in args.loss_type}
        train_losses["total_loss"] = 0.0
        for feature, label, _ in tqdm(train_dataset, total=len(train_dataset), desc="Traning Epoch", unit="batch", leave=False):
            if args.cpu:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()
            regular_lr = cosine_lr.get_regular_lr(iter_num)
            cosine_lr._set_lr(optimizer, regular_lr)

            prediction = model(input)
            optimizer.zero_grad()

            loss_values = []
            for loss_name, loss_func in losses.items():
                if loss_code[loss_name] == 0: continue
                loss_values.append(loss_func(prediction, target))
                train_losses[loss_name] += loss_values[-1].item()
            total_loss = sum(loss_values)
            train_losses["total_loss"] += total_loss.item()

            total_loss.backward()
            optimizer.step()
            iter_num += 1

        train_loss_curve.append(train_losses["total_loss"] / len(train_dataset))
        log_str = "[epoch: %d/%d] " % (epoch_num, args.num_epochs)

        for k in args.log_loss:
            v = train_losses[k]
            log_str += "| %s: %.4f " % (k, v / len(train_dataset))

        logger.info(log_str)
        if (epoch_num > 0 and epoch_num % args.save_freq == 0) or (epoch_num == args.num_epochs - 1):
            checkpoint(model, iter_num, args, logger)

        if epoch_num > 0 and epoch_num % args.eval_freq == 0 or (epoch_num == args.num_epochs - 1):
            if len(val_dataset) == 0: continue
            model.eval()
            with torch.no_grad():
                val_losses = {k: 0.0 for k in args.val_metric}
                val_losses["total_loss"] = 0.0
                for feature, label, _ in tqdm(val_dataset, total=len(val_dataset), desc="Val Epoch", unit="batch", leave=False):
                    if args.cpu:
                        input, target = feature, label
                    else:
                        input, target = feature.cuda(), label.cuda()
                    prediction = model(input)

                    for loss_name, loss_func in losses.items():
                        if loss_code[loss_name] == 0: continue
                        loss = loss_func(prediction, target)
                        val_losses["total_loss"] += loss.item()
                        if loss_name in val_metrics:
                            val_losses[loss_name] += loss.item()

                    for val_name, val_func in val_metrics.items():
                        if not val_name in losses:
                            val_losses[val_name] += val_func(prediction, target)

                log_str = "*** Eval [epoch: %d/%d] " % (epoch_num, args.num_epochs)
                for k, v in val_losses.items():
                    log_str += "| %s: %.4f " % (k, v / len(val_dataset))
                logger.info(log_str)

    logger.info("Training Done")
    ## plot train loss curve
    plt.plot(train_loss_curve)
    plt.savefig(os.path.join(args.res_root, "train_loss.png"))

if __name__ == "__main__":
    train()
