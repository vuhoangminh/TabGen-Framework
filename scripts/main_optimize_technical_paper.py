import os
import json
import time
import argparse
import subprocess
import numpy as np
from rich import print
import gc
from termcolor import colored

from hyperopt import hp, STATUS_OK
from rich import print

import engine.utils.hyperopt_utils as hyperopt_utils
import engine.utils.path_utils as path_utils

from engine.evaluate_technical_paper import (
    compute_statistical_metrics,
    compute_ml_metrics_all_ml_methods,
    compute_dp_metrics,
)

from engine.datasets import get_dataset
from engine.utils.data_utils import get_epochs_max_and_max_trials
from engine.config import config


# ===========================================================================
# objective
# ===========================================================================
def update_params(params):
    for key, value in params.items():
        if key in [
            "generator_lr",
            "generator_decay",
            "discriminator_lr",
            "discriminator_decay",
            "l2scale",
            "is_loss_corr",
            "is_loss_dwp",
        ]:
            params[key] = 10**value
    return params


def update_cmd(params, cmd, ignore_keys=[]):
    updated_cmd = cmd
    for key, value in params.items():
        if key in ignore_keys:
            continue
        updated_cmd += f" --{key} {value}"
    return updated_cmd


def get_folders_by_modified_time(directory, reverse=False):
    """
    Gets a list of folders in a directory sorted by modification time.

    Args:
      directory: The path to the directory.
      reverse: If True, sorts by newest first, otherwise oldest first.

    Returns:
      A list of folder paths sorted by modification time.
    """

    folders = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]
    folders.sort(key=os.path.getmtime, reverse=reverse)
    return folders


def _add_dict_to_args(parser, dictionary, is_overwrite=False):
    """Adds a dictionary of arguments to a copy of the argparse parser.

    Args:
      parser: The argparse parser object.
      dictionary: The dictionary of arguments to add.

    Returns:
      A new argparse parser with the added arguments.
    """
    import copy

    new_parser = copy.deepcopy(parser)
    for key, value in dictionary.items():
        new_parser.add_argument(
            f"--{key}", type=type(value), default=value, help=f"Value for {key}"
        )
    return new_parser


def add_dict_to_args(parser, dictionary):
    """Adds or overwrites arguments in the parser with values from a dictionary.

    Args:
        parser: The argparse parser object.
        dictionary: The dictionary of arguments and their values.

    Returns:
        The modified argparse parser object.  (No deep copy is made)
    """

    import copy

    new_parser = copy.deepcopy(parser)

    for key, value in dictionary.items():
        # Check if the argument already exists
        if any(action.dest == key for action in new_parser._actions):
            # Argument exists: Modify it
            for action in new_parser._actions:
                if action.dest == key:
                    # Found the action to modify
                    if isinstance(action, argparse._StoreTrueAction):  # Boolean flag
                        if value is True or value == "True":
                            action.default = (
                                True  # Directly update the action's default
                            )
                        elif value is False or value == "False":
                            action.default = False
                        elif value is None:  # optional flag
                            pass
                        else:
                            raise ValueError(
                                f"Invalid value for boolean flag --{key}: {value}"
                            )
                    else:  # Normal argument
                        action.default = value  # Directly update the action's default
                    break  # Stop searching once modified
        else:
            # Argument doesn't exist: Add it
            if type(value) is bool:  # Boolean flag
                if value is not None:
                    new_parser.add_argument(
                        f"--{key}", action="store_true", help=f"Value for {key}"
                    )
                else:  # optional boolean flag
                    new_parser.add_argument(
                        f"--{key}",
                        action="store_true",
                        default=False,
                        help=f"Value for {key}",
                    )
            else:
                new_parser.add_argument(
                    f"--{key}", type=type(value), default=value, help=f"Value for {key}"
                )

    return new_parser  # Return the modified parser


def objective(params):
    def construct_return_dict(
        loss,
        reason,
        params,
        df_score,
        df_score_ml,
        df_score_augment,
        df_score_dp,
        dir_logs,
    ):
        return {
            "loss": loss,
            "status": STATUS_OK,
            "reason": reason,
            "params": params,
            "scores_statistics": (
                df_score.iloc[0].to_dict() if df_score is not None else {}
            ),
            "scores_ml": (
                df_score_ml.iloc[0].to_dict() if df_score_ml is not None else {}
            ),
            "scores_ml_augment": (
                df_score_augment.iloc[0].to_dict()
                if df_score_augment is not None
                else {}
            ),
            "scores_dp": (
                df_score_dp.iloc[0].to_dict() if df_score_dp is not None else {}
            ),
            "dir_logs": dir_logs if "dir_logs" in locals() else None,
        }

    params = update_params(params)
    print(params)

    try:
        if args.row_number is not None:
            _cmd = f"python -W ignore scripts/main_technical_paper.py --dir_logs {args.dir_logs} --is_test {args.is_test} --dataset {args.dataset} --arch {args.arch} --loss_version {args.loss_version} --checkpoint_freq 100 --is_condvec {args.is_condvec} --row_number {args.row_number}"
        else:
            _cmd = f"python -W ignore scripts/main_technical_paper.py --dir_logs {args.dir_logs} --is_test {args.is_test} --dataset {args.dataset} --arch {args.arch} --loss_version {args.loss_version} --checkpoint_freq 100 --is_condvec {args.is_condvec}"

        cmd = update_cmd(params, _cmd)

        new_parser = add_dict_to_args(parser, params)
        new_args = new_parser.parse_args()
        dir_logs = os.path.join(
            args.dir_logs, path_utils.get_folder_technical_paper(new_args)
        )
        print(f">> logging to {dir_logs}")

        print(f">> running {cmd}")

        # os.system(cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # If you need to print or use the output, use result.stdout.decode()
        print(result.stdout.decode())

        folder = path_utils.get_filename(dir_logs)
        D = get_dataset(new_args.dataset)
        discrete_columns = D.discrete_columns
        continuous_columns = [
            c for c in list(D.data_train) if c not in discrete_columns
        ]

        loss = 0  # we will update later
        df_score = compute_statistical_metrics(
            None,
            folder,
            discrete_columns,
            continuous_columns,
            mode="last",
        )
        df_score = df_score.drop(df_score.columns[:6], axis=1)  # Drop first 6 columns
        df_score = df_score.drop(df_score.columns[-2:], axis=1)  # Drop last 2 columns

        df_score_ml = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode="last",
            task="single",
        )
        df_score_ml = df_score_ml.drop(df_score_ml.columns[:6], axis=1)

        df_score_augment = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode="last",
            task="augment",
        )
        df_score_augment = df_score_augment.drop(df_score_augment.columns[:6], axis=1)

        if args.module != "public":
            df_score_dp = compute_dp_metrics(
                D,
                None,
                folder,
                key_fields=D.key_fields,
                sensitive_fields=D.sensitive_fields,
            )
            df_score_dp = df_score_dp.drop(df_score_dp.columns[:6], axis=1)
        else:
            df_score_dp = None

        # Find best loss
        reason = "success"

        gc.collect()

        return construct_return_dict(
            loss,
            reason,
            params,
            df_score,
            df_score_ml,
            df_score_augment,
            df_score_dp,
            dir_logs,
        )

    except subprocess.CalledProcessError as e:
        reason = f"Command '{cmd}' failed with return code {e.returncode}"
        print(reason)
        print(e.stderr.decode())
        loss = np.inf

    except Exception as e:
        reason = str(e)
        loss = np.inf

    except RuntimeError as e:  # synthetic tensor contains nan values
        reason = str(e)
        loss = np.inf

    return construct_return_dict(loss, reason, params, None, None, None, None, None)


def init_search_space(args):
    try:
        epochs_max, max_trials = get_epochs_max_and_max_trials(
            args.dataset, config.DICT_DATASETS
        )
    except:
        epochs_max, max_trials = 2000, 30

    batch_size_choices = list(range(500, 30000 + 1, 500))
    batch_size_choices_ctab = list(range(500, 4000 + 1, 500))  # OOM
    epochs_choices = list(range(100, epochs_max + 1, 100))
    epochs_choices_big_datasets_ctab = list(range(100, 401, 100))

    if args.is_test:
        search_space = {"epochs": hp.choice("epochs", [100])}

        if args.arch == "ctab":
            search_space["batch_size"] = hp.choice("batch_size", [1000])
        else:
            search_space["batch_size"] = hp.choice("batch_size", [1000])
    else:
        if args.arch == "ctab" and args.dataset == "mnist28":
            search_space = {
                "epochs": hp.choice("epochs", epochs_choices_big_datasets_ctab)
            }
        else:
            search_space = {"epochs": hp.choice("epochs", epochs_choices)}

        if args.arch == "ctab":
            search_space["batch_size"] = hp.choice(
                "batch_size", batch_size_choices_ctab
            )
        else:
            search_space["batch_size"] = hp.choice("batch_size", batch_size_choices)

    if args.arch in ["ctgan", "copulagan", "dpcgans"]:
        search_space.update(
            {
                "embedding_dim": hp.choice("embedding_dim", [32, 64, 128, 256]),
                "generator_dim": hp.choice("generator_dim", [32, 64, 128, 256]),
                "discriminator_dim": hp.choice("discriminator_dim", [32, 64, 128, 256]),
                "generator_lr": hp.uniform("generator_lr", -5, -3),
                "generator_decay": hp.uniform("generator_decay", -7, -5),
                "discriminator_lr": hp.uniform("discriminator_lr", -5, -3),
                "discriminator_decay": hp.uniform("discriminator_decay", -7, -5),
            }
        )

        if args.module in ["public", "gm"]:
            search_space.update(
                {
                    "private": hp.choice("private", [0]),
                    "dp_sigma": hp.uniform("dp_sigma", 0.00001, 1),
                    "dp_weight_clip": hp.uniform("dp_weight_clip", 0.01, 2),
                }
            )
        else:
            search_space.update(
                {
                    "private": hp.choice("private", [0, 1]),
                    "dp_sigma": hp.uniform("dp_sigma", 0.00001, 1),
                    "dp_weight_clip": hp.uniform("dp_weight_clip", 0.001, 2),
                }
            )

    elif args.arch in ["tvae"]:
        search_space.update(
            {
                "embedding_dim": hp.choice("embedding_dim", [32, 64, 128, 256]),
                "compress_dims": hp.choice("compress_dims", [32, 64, 128, 256]),
                "decompress_dims": hp.choice("decompress_dims", [32, 64, 128, 256]),
                "loss_factor": hp.choice("loss_factor", [0.25, 0.5, 1, 2, 4]),
                "l2scale": hp.uniform("l2scale", -6, -4),
            }
        )
    elif args.arch in ["ctab"]:
        search_space.update(
            {
                "test_ratio": hp.choice("test_ratio", [0.1, 0.2, 0.3, 0.4, 0.5]),
                "n_class_layer": hp.choice("n_class_layer", [1, 2, 3, 4]),
                "class_dim": hp.choice("class_dim", [32, 64, 128, 256]),
                "random_dim": hp.choice("random_dim", [16, 32, 64, 128]),
                "num_channels": hp.choice("num_channels", [16, 32, 64]),
            }
        )

    if args.loss_version != 0:
        search_space.update(
            {
                "is_loss_corr": hp.uniform("is_loss_corr", -2, 6),
                "is_loss_dwp": hp.uniform("is_loss_dwp", -10, 1),
                "n_moment_loss_dwp": hp.choice("n_moment_loss_dwp", [1, 2, 3, 4]),
            }
        )

    return search_space


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir_logs",
        type=str,
        default="database/gan_optimize/",
        help="dir logs",
    )
    parser.add_argument(
        "--is_test",
        default=1,
        # default=0,
        type=int,
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--dataset",
        # default="abalone",
        default="higgs-small",
        # "adult",
        # "mnist12",
        # "mnist28",
        # "news",
        # "diabetes",
        # "diabetesbalanced",
        # "house",
        # "covertype",
        # "credit",
        # "intrusion",
        # "abalone",
        # "buddy",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="ctgan",
        # "ctgan",
        # "copulagan",
        # "tvae",
        # "dpcgans",
        # "ctab",
    )
    parser.add_argument(
        "--loss_version",
        default=2,
        choices=[
            0,  # conventional
            2,  # version 2
            4,  # version 4: only correlation loss
            5,  # version 5: only distribution loss
        ],
        type=int,
    )
    parser.add_argument(
        "--is_condvec",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--bo_method",
        default="ior",
        # default="sbo",
        choices=["ior", "sbo"],
    )
    parser.add_argument(
        "--bo_method_agg",
        # default="mean",
        default="median",
        choices=["mean", "median"],
    )
    parser.add_argument(
        "--module",
        default="public",
        choices=[
            "public",
            "gm",
            "dp",
            "gmdp",
        ],  # dp and gmdp are only for sensitive data
    )

    # generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
    parser.add_argument(
        "--row_number",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    # loop indefinitely and stop whenever you like
    max_successful_trials = args.max_trials

    # init search space
    search_space = init_search_space(args)
    print(search_space)

    # dp for biobank
    if args.module in ["public", "gm"]:
        evaluations = ["statistics", "ml", "ml_augment"]
    elif args.module == "dp":
        evaluations = ["dp"]
    elif args.module == "gmdp":
        evaluations = ["statistics", "ml", "ml_augment", "dp"]

    if args.module == "public":
        module = ""
    else:
        module = f"_module-{args.module}"

    # hyperopt
    database_path = "database"

    if args.row_number is not None:
        filename = f"{args.dataset}_rownum-{args.row_number}_{args.arch}_loss_version-{args.loss_version}-{args.is_condvec}{module}"
    else:
        filename = f"{args.dataset}_{args.arch}_loss_version-{args.loss_version}-{args.is_condvec}{module}"

    if args.is_test:
        filename = f"test_" + filename

    if args.bo_method == "ior":
        folder = "optimization"
    elif args.bo_method == "sbo":
        folder = f"optimization_sbo_{args.bo_method_agg}"
    hyperopt_project_path = path_utils.get_hyperopt_path(
        filename, database_path=database_path, folder=folder
    )

    trials = hyperopt_utils.load_project(hyperopt_project_path)
    algo = "tpe"
    is_continue = True

    if args.bo_method == "ior":
        I = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
            hyperopt_project_path,
        )
    else:
        agg = args.bo_method_agg
        I = hyperopt_utils.StandardObjectiveOptimizationGenerativeModel(
            hyperopt_project_path,
            agg=agg,
        )

    if len(trials.trials) > 0:
        I.update_trials_losses(evaluations=evaluations)

        n_successful_trials = hyperopt_utils.get_number_successful_trials(
            trials, success_value="success"
        )
        n_trials = hyperopt_utils.get_number_trials(trials)
        if n_successful_trials >= max_successful_trials or (
            n_successful_trials == 0 and n_trials >= args.max_trials
        ):
            is_continue = False

    while is_continue:
        print(colored("=" * 100, "red"))
        n_trials, n_successful_trials = hyperopt_utils.run_trials(
            project_path=hyperopt_project_path,
            objective=objective,
            space=search_space,
            algo=algo,
        )
        I.update_trials_losses(evaluations=evaluations)

        if n_successful_trials >= max_successful_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done", "red"))
            print(colored("=" * 100, "red"))
        if n_successful_trials == 0 and n_trials >= args.max_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done but n_successful_trials = 0", "red"))
            print(colored("=" * 100, "red"))
