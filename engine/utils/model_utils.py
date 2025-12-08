import os
import numpy as np
import pandas as pd
import torch
import shutil
import engine.logger as logger
from engine.utils.eval_utils import (
    compute_dwp,
    compute_diff_correlation,
)


def get_epoch_best_model(folder, weight=10):
    # weight = 2.82
    # weight = 10

    score_path = f"{folder}/score.csv"

    if os.path.isfile(score_path):
        df_score = pd.read_csv(score_path, sep="\t", header=0, index_col=0)
    else:
        df_score = None

    df = pd.read_csv(f"{folder}/preprocessed.csv", sep="\t", header=0, index_col=0)

    dmin, corrmin, dwpmin, epochmin = np.inf, 0, 0, 0
    list_i_epoch, list_corr, list_dwp, list_d = [], [], [], []
    for i_epoch in [50 * (i + 1) for i in range(60)]:
        filename = f"fake_{i_epoch:05}.csv"

        try:
            r = df_score.loc[df_score["i_epoch"] == i_epoch]
            corr = r["corr"].values[0]
            dwp = r["dwp"].values[0]
        except:
            try:
                df_fake = pd.read_csv(
                    f"{folder}/{filename}",
                    sep="\t",
                    header=0,
                    index_col=0,
                )
                corr = compute_diff_correlation(df, df_fake)
                dwp, _, _ = compute_dwp(df, df_fake)
            except:
                corr, dwp = None, None

        if corr is not None:
            d = corr + dwp * weight
            if dmin > d:
                dmin = d
                epochmin = i_epoch
                corrmin = corr
                dwpmin = dwp

            list_i_epoch.append(i_epoch)
            list_corr.append(corr)
            list_dwp.append(dwp)
            list_d.append(d)

    return epochmin


def save_checkpoint(
    info,
    model,
    optim,
    dir_logs,
    save_model,
    i_epoch,
    save_all_from=None,
    is_best=True,
    suffix="generator",
):
    os.system("mkdir -p " + dir_logs)
    if save_all_from is None:
        path_ckpt_info = os.path.join(
            dir_logs, f"ckpt_info_{i_epoch:05}_{suffix}.pth.tar"
        )
        path_ckpt_model = os.path.join(
            dir_logs, f"ckpt_model_{i_epoch:05}_{suffix}.pth.tar"
        )
        path_ckpt_optim = os.path.join(
            dir_logs, f"ckpt_optim_{i_epoch:05}_{suffix}.pth.tar"
        )
        path_best_info = os.path.join(dir_logs, f"best_info_{suffix}.pth.tar")
        path_best_model = os.path.join(dir_logs, f"best_model_{suffix}.pth.tar")
        path_best_optim = os.path.join(dir_logs, f"best_optim_{suffix}.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        # info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
    else:
        is_best = False  # because we don't know the test accuracy
        path_ckpt_info = os.path.join(dir_logs, "ckpt_epoch,{}_info_{suffix}.pth.tar")
        path_ckpt_model = os.path.join(dir_logs, "ckpt_epoch,{}_model_{suffix}.pth.tar")
        path_ckpt_optim = os.path.join(dir_logs, "ckpt_epoch,{}_optim_{suffix}.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info["epoch"]))
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info["epoch"]))
            torch.save(optim, path_ckpt_optim.format(info["epoch"]))
        if info["epoch"] > 1 and info["epoch"] < save_all_from + 1:
            os.system("rm " + path_ckpt_info.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_model.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_optim.format(info["epoch"] - 1))
    if not save_model:
        print("Warning train.py: checkpoint not saved")


def get_saved_model(dir, suffix):
    files = []
    for file in os.listdir(dir):
        if f"_{suffix}.pth.tar" in file and "ckpt_optim_" in file:
            files.append(file)
    files.sort()
    last_file = files[-1]
    i_epoch = last_file.replace("ckpt_optim_", "").replace(f"_{suffix}.pth.tar", "")
    return i_epoch


def load_checkpoint(model, optimizer, path_ckpt, suffix="generator", i_epoch=None):
    is_loadable = True
    start_epoch = 0
    best_metric = 0
    exp_logger = None

    if i_epoch is None:
        i_epoch = get_saved_model(path_ckpt, suffix)
    else:
        i_epoch = f"{i_epoch:05}"

    path_ckpt_info = os.path.join(path_ckpt, f"ckpt_info_{i_epoch}_{suffix}.pth.tar")
    path_ckpt_model = os.path.join(path_ckpt, f"ckpt_model_{i_epoch}_{suffix}.pth.tar")
    path_ckpt_optim = os.path.join(path_ckpt, f"ckpt_optim_{i_epoch}_{suffix}.pth.tar")
    print("---------------------------------------------")
    print(path_ckpt_info)
    print(path_ckpt_model)
    print(path_ckpt_optim)
    print("---------------------------------------------")
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_metric = 0
        exp_logger = None
        if "epoch" in info:
            start_epoch = info["epoch"]
        else:
            print("Warning train.py: no epoch to resume")
        if "best_acc1" in info:
            best_metric = info["best_acc1"]
        elif "best_metric" in info:
            best_metric = info["best_metric"]
        else:
            print("Warning train.py: no best_metric to resume")
        if "exp_logger" in info:
            exp_logger = info["exp_logger"]
        else:
            print("Warning train.py: no exp_logger to resume")
    else:
        print(
            "Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info)
        )
        is_loadable = False
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print(
            "Warning train.py: no model checkpoint found at '{}'".format(
                path_ckpt_model
            )
        )
        is_loadable = False
    if optimizer is not None and os.path.isfile(path_ckpt_optim):
        optim_state = torch.load(path_ckpt_optim)
        optimizer.load_state_dict(optim_state)
    else:
        print(
            "Warning train.py: no optim checkpoint found at '{}'".format(
                path_ckpt_optim
            )
        )
        is_loadable = False
    print(
        "=> loaded checkpoint '{}' (epoch {}, best_metric {})".format(
            path_ckpt, start_epoch, best_metric
        )
    )
    start_epoch += 1
    return start_epoch, best_metric, exp_logger, is_loadable


def flatten(d, parent_key="", sep="_"):
    import collections

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def make_meters():
    meters_dict = {
        "loss_d": logger.AvgMeter(),
        "loss_g": logger.AvgMeter(),
        "epoch_time": logger.ValueMeter(),
    }
    return meters_dict


def make_meters_ctgan():
    meters_dict = {
        "loss_d": logger.AvgMeter(),
        "loss_g": logger.AvgMeter(),
        "loss_corr": logger.AvgMeter(),
        "loss_dwp": logger.AvgMeter(),
        "metric_corr": logger.AvgMeter(),
        "metric_dwp": logger.AvgMeter(),
        "epoch_time": logger.ValueMeter(),
        "dp_sigma": logger.ValueMeter(),
        "dp_weight_clip": logger.ValueMeter(),
        "dp_epsilon": logger.ValueMeter(),
        "dp_delta": logger.ValueMeter(),
        "dp_opt_order": logger.ValueMeter(),
    }
    return meters_dict


def make_meters_tvae():
    meters_dict = {
        "loss_1": logger.AvgMeter(),
        "loss_2": logger.AvgMeter(),
        "loss_corr": logger.AvgMeter(),
        "loss_dwp": logger.AvgMeter(),
        "metric_corr": logger.AvgMeter(),
        "metric_dwp": logger.AvgMeter(),
        "epoch_time": logger.ValueMeter(),
    }
    return meters_dict
