import os
import re
import math
import random
import numpy as np
import pandas as pd
import argparse
from rich import print

import engine.config as config
from engine.datasets import get_dataset, AdultDataset
import engine.utils.path_utils as path_utils
from engine.utils.hyperopt_utils import (
    get_best_set_params,
    get_best_set_params_imbalanced,
)

from engine.utils.eval_utils import *
from engine.utils.eval_dp_utils import *
from engine.experiment_technical_paper import (
    perform_xgboost,
    perform_linear_regression,
    perform_bagging,
    perform_randomforest,
    perform_svm,
)


def get_loss_version(folder):
    # adult-copulagan-bs_16000-epochs_3000-ed_128-moment_1-losscorcorr_0-lossdis_1-condvec_0
    # adult-copulagan-bs_16000-epochs_3000-ed_128-moment_1-losscorcorr_0-normalizedlossdis_1-condvec_0
    if "moment" in folder and "normalizedlossdis" not in folder:  # version 2
        return 2
    elif "moment" in folder and "normalizedlossdis" in folder:  # version 3
        return 3
    else:
        return 1


def init_score(folder, filename):
    def extract_before_after(s):
        # Define a pattern to match "lv_" followed by any integer
        pattern = r"(.*?)-(lv_\d+)-(.*)"
        match = re.match(pattern, s)

        if match:
            before_lv = match.group(1)
            lv_value = match.group(2)
            after_lv = match.group(3)
            return before_lv, lv_value, after_lv
        else:
            return None, None, None

    def _extract_dataset_model(s):
        # Check for 'test-' prefix and adjust the input string
        if s.startswith("test-"):
            s = s[len("test-") :]

        match = re.match(r"(.*?)(-)([^-]+)$", s)

        if "rownum_" in match.group(1):
            match_dataset = re.match(r"([a-zA-Z0-9_]+)-rownum_\d+", match.group(1))
        else:
            match_dataset = re.match(r"([a-zA-Z0-9_]+)", match.group(1))

        dataset = match_dataset.group(1) if match_dataset else None

        if dataset.endswith("-"):
            dataset = dataset[:-1]  # Remove the trailing "-"
        model = match.group(3)

        return dataset, model

    def extract_dataset_model(s):
        # Check for 'test-' prefix and adjust the input string
        if s.startswith("test-"):
            s = s[len("test-") :]

        model = s.split("-", 1)[-1]
        last_dash_index = s.rfind("-")
        s = s[:last_dash_index]

        if "rownum" in s:
            last_dash_index = s.rfind("-")
            dataset = s[:last_dash_index]
        else:
            dataset = s

        return dataset, model

    before_lv, lv_value, after_lv = extract_before_after(folder)
    dataset, model = extract_dataset_model(before_lv)

    if lv_value == "lv_0":
        match = re.match(r".*-condvec_(\d+)", after_lv)

        condvec = match.group(1)  # "0"
        score = {
            "folder": folder,
            "filename": filename,
            "model": model,
            "losscorr": 0,
            "lossdwp": 0,
            "condvec": condvec,
        }
    elif lv_value in ["lv_2", "lv_4", "lv_5"]:
        match = re.match(
            r".*losscorcorr_([\d.e+-]+)-lossdis_([\d.e+-]+)-condvec_(\d+)",
            after_lv,
        )

        losscorr = match.group(1)
        lossdis = match.group(2)
        condvec = match.group(3)
        score = {
            "folder": folder,
            "filename": filename,
            "model": model,
            "losscorr": losscorr,
            "lossdwp": lossdis,
            "condvec": condvec,
        }
    else:
        raise NotImplementedError

    return score, dataset


def compute_statistical_metrics(
    df_score,
    folder,
    discrete_columns,
    continuous_columns,
    mode="optimal",
    df_real=None,
    df_fake=None,
):
    dir_logs = os.path.join(f"database/gan_optimize/{folder}")

    if df_real is None:
        df_real = pd.read_csv(
            os.path.join(dir_logs, "preprocessed.csv"),
            sep="\t",
            header=0,
            index_col=0,
        )
    # print(df)

    if mode == "optimal":
        filename = config.LIST_BEST[folder]
    else:
        _, filename = path_utils.find_non_largest_csv_files(dir_logs)

    if df_fake is None:
        df_fake = pd.read_csv(
            f"database/gan_optimize/{folder}/{path_utils.get_filename(filename)}",
            sep="\t",
            header=0,
            index_col=0,
        )
    # print(df_fake)

    # update discrete_columns based on sampled rows and columns
    discrete_columns = [col for col in discrete_columns if col in df_real.columns]
    continuous_columns = [col for col in continuous_columns if col in df_real.columns]

    cramer, note_cramer = compute_cramers_v_correlation(
        df_real.copy(),
        df_fake.copy(),
        cols=discrete_columns,
    )

    pearson, note_pearson_continuous = compute_pearson_correlation(
        df_real.copy(),
        df_fake.copy(),
        cols=continuous_columns,
    )

    score, _ = init_score(folder, filename)

    score.update(
        {
            "kl_divergence_discrete": compute_kl_divergence(
                df_real.copy(),
                df_fake.copy(),
                cols=discrete_columns,
                normalize=False,
                handle_missing=False,
                is_continuous=False,
            ),
            "kl_divergence_continuous": compute_kl_divergence(
                df_real.copy(),
                df_fake.copy(),
                cols=continuous_columns,
                normalize=False,
                handle_missing=True,
                is_continuous=True,
            ),
            "chisquare_discrete": compute_chisquare_test(
                df_real.copy(),
                df_fake.copy(),
                cols=discrete_columns,
                handle_missing=False,
            ),
            "kolmogorov_smirnov_continuous": compute_kolmogorov_smirnov_test(
                df_real.copy(),
                df_fake.copy(),
                cols=continuous_columns,
                handle_missing=False,
            ),
            "cramer_discrete": cramer,
            "pearson_continuous": pearson,
            "dwp_discrete": compute_dwp(
                df_real.copy(),
                df_fake.copy(),
                discrete_columns=discrete_columns,
                is_included_discrete=True,
                is_included_continuous=False,
            )[0],
            "dwp_continuous": compute_dwp(
                df_real.copy(),
                df_fake.copy(),
                discrete_columns=discrete_columns,
                is_included_discrete=False,
                is_included_continuous=True,
            )[0],
            "cramer_discrete_note": note_cramer,
            "pearson_continuous_note": note_pearson_continuous,
        }
    )

    print(score)

    if df_score is None:
        df_score = pd.DataFrame.from_dict(score, orient="index").T
    else:
        score = pd.DataFrame.from_dict([score])
        df_score = pd.concat([df_score, score], ignore_index=True)

    print(df_score)

    return df_score


def compute_ml_metrics_one_ml_method(
    function, X, y, X_fake, y_fake, X_test, y_test, task, output
):
    def f(x, y, task):
        if task == "single":
            return abs(x - y)  # lower is better
        else:
            return y - x  # higher is better except for regression (mae + mse)

    if output == "classification":
        acc, precision, recall, f1, gmean, roc = tuple(
            f(x, y, task)
            for x, y in zip(
                function(X, y, X_test, y_test, output),
                function(X_fake, y_fake, X_test, y_test, output),
            )
        )
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gmean": gmean,
            "roc": roc,
        }
    else:
        mae, mse, r2 = tuple(
            f(x, y, task)
            for x, y in zip(
                function(X, y, X_test, y_test, output),
                function(X_fake, y_fake, X_test, y_test, output),
            )
        )
        return {
            "mae": mae,
            "mse": mse,
            "r2": r2,
        }


def compute_ml_metrics_all_ml_methods(
    D,
    df_score,
    folder,
    mode="last",
    task="single",  # single, augment
    ml_params_setting={
        "mode": "best",
        "path": "database/optimization_ml_method",
    },
    df_fake=None,
):
    def f(x, y, task):
        if task == "single":
            return abs(x - y)  # lower is better
        else:
            return y - x  # higher is better except for regression (mae + mse)

    dir_logs = os.path.join(f"database/gan_optimize/{folder}")

    if mode == "optimal":
        filename = config.LIST_BEST[folder]
    else:
        _, filename = path_utils.find_non_largest_csv_files(dir_logs)

    if df_fake is None:
        df_fake = pd.read_csv(
            f"database/gan_optimize/{folder}/{path_utils.get_filename(filename)}",
            sep="\t",
            header=0,
            index_col=0,
        )

    # update discrete_columns based on sampled rows and columns
    features_columns = [col for col in D.features if col in df_fake.columns]

    X = D.data_train[features_columns]
    y = D.data_train[D.target]
    X_fake = df_fake[features_columns]
    y_fake = df_fake[D.target]

    if task == "augment":
        X_fake = pd.concat([X_fake, X], ignore_index=True)
        y_fake = pd.concat([y_fake, y], ignore_index=True)

    X_test = D.data_test[features_columns]
    y_test = D.data_test[D.target]

    dict_method = {
        "regression": perform_linear_regression,
        "svm": perform_svm,
        "randomforest": perform_randomforest,
        "bagging": perform_bagging,
        "xgboost": perform_xgboost,
    }

    score, dataset = init_score(folder, filename)

    for method, function in dict_method.items():
        if ml_params_setting["mode"] == "best":
            # Added by the author
            fmin = get_best_set_params(
                f'{ml_params_setting["path"]}/{dataset}_{method}.hyperopt'
            )
            # Added by the author
            # we deal with imbalanced dataset in phase 3 here

            best_params = fmin["params"]
            if "device" in fmin:
                device = fmin["device"]
            else:
                device = "gpu"

        else:
            best_params = {}

        if D.output == "classification":
            acc, precision, recall, f1, gmean, roc = tuple(
                f(x, y, task)
                for x, y in zip(
                    function(
                        X,
                        y,
                        X_test,
                        y_test,
                        D.output,
                        params=best_params,
                        device=device,
                    ),
                    function(
                        X_fake,
                        y_fake,
                        X_test,
                        y_test,
                        D.output,
                        params=best_params,
                        device=device,
                    ),
                )
            )
            score[f"{method}_acc"] = acc
            score[f"{method}_precision"] = precision
            score[f"{method}_recall"] = recall
            score[f"{method}_f1"] = f1
            score[f"{method}_gmean"] = gmean
            score[f"{method}_roc"] = roc

        else:
            mae, mse, r2 = tuple(
                f(x, y, task)
                for x, y in zip(
                    function(
                        X,
                        y,
                        X_test,
                        y_test,
                        D.output,
                        params=best_params,
                        device=device,
                    ),
                    function(
                        X_fake,
                        y_fake,
                        X_test,
                        y_test,
                        D.output,
                        params=best_params,
                        device=device,
                    ),
                )
            )

            score[f"{method}_mae"] = mae
            score[f"{method}_mse"] = mse
            score[f"{method}_r2"] = r2

    print(score)

    if df_score is None:
        df_score = pd.DataFrame.from_dict(score, orient="index").T
    else:
        score = pd.DataFrame.from_dict([score])
        df_score = pd.concat([df_score, score], ignore_index=True)

    print(df_score)

    return df_score


def compute_dp_metrics(
    D,
    df_score,
    folder,
    key_fields,
    sensitive_fields,
    df_fake=None,
):
    dir_logs = os.path.join(f"database/gan_optimize/{folder}")
    _, filename = path_utils.find_non_largest_csv_files(dir_logs)
    scores, dataset = init_score(folder, filename)

    df_hold = D.data_test
    df_real = D.data_train

    if df_fake is None:
        df_fake = pd.read_csv(
            f"database/gan_optimize/{folder}/{path_utils.get_filename(filename)}",
            sep="\t",
            header=0,
            index_col=0,
        )

    # update discrete_columns based on sampled rows and columns
    key_fields = [col for col in key_fields if col in df_real.columns]
    sensitive_fields = [col for col in sensitive_fields if col in df_real.columns]

    # Iterate through the columns of df1 and set the dtypes in df2
    for column in df_fake.columns:
        if column in df_real.columns:  # Check if the column exists in df2
            df_real[column] = df_real[column].astype(df_fake[column].dtype)
            df_hold[column] = df_hold[column].astype(df_fake[column].dtype)

    df_test = df_hold.copy()

    # Step 1: k-anonymization
    start = time.time()
    x = compute_k_anonymization(df_real, df_fake, key_fields)
    print(f"k-anonymization: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 2: l-Diversity
    start = time.time()
    x = compute_l_diversity_distinct(df_real, df_fake, key_fields)
    print(f"l-Diversity: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 3: k-map
    start = time.time()
    x = compute_k_map(df_real, df_fake, key_fields)
    print(f"k-map: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 4: Delta presence
    start = time.time()
    x = compute_delta_presence(df_real, df_fake, key_fields)
    print(f"delta_presence: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 5: Evaluate re-identification
    start = time.time()
    x = compute_re_identification(df_real, df_fake)
    print(f"re_identification: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 6: Domiasmia
    start = time.time()
    x = compute_domiasmia(df_test, df_fake, df_real, df_fake.sample(frac=0.2), "prior")
    print(f"domiasmia prior: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 7: Categorical CAP
    start = time.time()
    x = compute_categoricalcap(df_real, df_fake, key_fields, sensitive_fields)
    print(f"categoricalcap: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 8: DCR Nndr
    start = time.time()
    x = bootstrap_compute_dcr_nndr(df_real, df_fake, data_percent=15, num_bootstrap=100)
    print(f"dcr nndr: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 9: Single-out risk (univariate)
    start = time.time()
    x = compute_single_out_risk(
        df_real,
        df_fake,
        df_hold,
    )
    print(f"single_out_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 11: Linkability risk
    start = time.time()
    x = compute_linkability_risk(
        df_real,
        df_fake,
        df_hold,
        aux_cols=(key_fields, sensitive_fields),
        n_attacks=len(df_hold.index),
    )
    print(f"linkability_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    # Step 12: Inference risk
    start = time.time()
    x = compute_inference_risk(
        df_real,
        df_fake,
        df_hold,
        key_fields,
        sensitive_fields,
        n_attacks=len(df_hold.index),
    )
    print(f"inference_risk: {x}, Time: {time.time() - start:.2f} seconds")
    scores.update(x)

    print(scores)

    if df_score is None:
        df_score = pd.DataFrame.from_dict(scores, orient="index").T
    else:
        scores = pd.DataFrame.from_dict([scores])
        df_score = pd.concat([df_score, scores], ignore_index=True)

    print(df_score)

    return df_score


# ===========================================================================
# main
# ===========================================================================
def run(mode, args, sort=True):
    LIST_DATASETS = [
        # small
        "adult",
        "diabetes",
        "diabetesbalanced",
        "news",
        "house",
        # large
        "covertype",
        "credit",
        "intrusion",
        "mnist12",
        "mnist28",
    ]

    if args.group_dataset == 0:
        list_datasets = [args.dataset]
    elif args.group_dataset == 1:
        list_datasets = [
            "adult",
            "diabetes",
            "news",
            "covertype",
        ]
    elif args.group_dataset == 2:
        list_datasets = [
            "house",
            "diabetesbalanced",
            "credit",
            "mnist12",
        ]
    elif args.group_dataset == 3:
        list_datasets = [
            "intrusion",
            "mnist28",
        ]
    else:
        list_datasets = LIST_DATASETS

    # random.shuffle(list_datasets)
    print(list_datasets)

    if not sort:
        random.shuffle(list_datasets)

    for dataset in list_datasets:
        print(f">> {dataset}")
        D = get_dataset(dataset, args.arch)
        discrete_columns = D.discrete_columns
        continuous_columns = [
            c for c in list(D.data_train) if c not in discrete_columns
        ]
        print(discrete_columns)
        print(continuous_columns)

        # directory
        dir = "database/evaluation"
        dir = dir + f"/{mode}"
        path_utils.make_dir(dir)

        path = os.path.join(dir, f"{dataset}.csv")
        path_ml = os.path.join(dir, f"{dataset}_ml.csv")
        path_ml_augment = os.path.join(dir, f"{dataset}_ml_augment.csv")

        # folders
        folders = os.listdir("database/gan")
        folders.sort()
        eval_folders = [folder for folder in folders if f"{dataset}-" in folder]
        # random.shuffle(eval_folders)
        # print(eval_folders)

        if os.path.exists(path):
            df_score = pd.read_csv(path, sep="\t", header=0, index_col=0)
        else:
            df_score = None

        if os.path.exists(path_ml):
            df_score_ml = pd.read_csv(path_ml, sep="\t", header=0, index_col=0)
        else:
            df_score_ml = None

        if os.path.exists(path_ml_augment):
            df_score_augment = pd.read_csv(
                path_ml_augment, sep="\t", header=0, index_col=0
            )
        else:
            df_score_augment = None

        # metrics
        for folder in eval_folders:
            if df_score is not None and folder in df_score["folder"].values:
                print(f"skipping {folder}")
                continue
            else:
                print(f">> {folder}")
                df_score = compute_statistical_metrics(
                    df_score,
                    folder,
                    discrete_columns,
                    continuous_columns,
                    mode=mode,
                )
                df_score.to_csv(path, sep="\t", encoding="utf-8")

        # ML performance
        for folder in eval_folders:
            if df_score_ml is not None and folder in df_score_ml["folder"].values:
                print(f"skipping {folder}")
                continue
            else:
                df_score_ml = compute_ml_metrics_all_ml_methods(
                    D,
                    df_score_ml,
                    folder,
                    mode=mode,
                    task="single",
                )
                df_score_ml.to_csv(path_ml, sep="\t", encoding="utf-8")

        # ML augmentation performance
        for folder in eval_folders:
            if (
                df_score_augment is not None
                and folder in df_score_augment["folder"].values
            ):
                print(f"skipping {folder}")
                continue
            else:
                df_score_augment = compute_ml_metrics_all_ml_methods(
                    D,
                    df_score_augment,
                    folder,
                    mode=mode,
                    task="augment",
                )
                df_score_augment.to_csv(path_ml_augment, sep="\t", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--dir_logs",
        type=str,
        default="database/gan_optimize/",
        help="dir logs",
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="ctgan",
        choices=["ctgan", "tvae", "copulagan", "dpcgans", "ctab"],
    )
    parser.add_argument(
        "--dataset",
        default="adult",
        # default="diabetesbalanced",
        # default="diabetes",
        # default="house",
        # default="news",
        # default="credit",
        # default="mnist12",
        # default="mnist28",
        # default="covertype",
        # default="intrusion",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--group_dataset",
        default=4,
        type=int,
        choices=[0, 1, 2, 3, 4],
    )

    # Parse the arguments
    args = parser.parse_args()
    for mode in ["last", "optimal"]:
        run(mode, args)


def test(mode="last"):
    eval_folders = [
        # "abalone-copulagan-lv_0-bs_500-epochs_300-ed_128-dd_64-gd_128-glr_1.84e-04-condvec_0"
        # "abalone-copulagan-lv_0-bs_7000-epochs_1300-ed_64-dd_128-gd_32-glr_4.96e-04-condvec_0"
        # "abalone-copulagan-lv_2-bs_10500-epochs_2000-ed_256-dd_32-gd_128-glr_5.61e-04-moment_2-losscorcorr_7.77e+03-lossdis_2.82e-07-condvec_0",
        # "buddy-copulagan-lv_0-bs_2000-epochs_1400-ed_256-dd_64-gd_32-glr_5.90e-04-condvec_1",
        # "churn2-copulagan-lv_0-bs_21000-epochs_1500-ed_32-dd_32-gd_256-glr_1.06e-05-condvec_0",
        "adult-tabddpm-lv_2-bs_256-epochs_20000-df_7-dm_10-dl_10-nl_3-lr_1.62e-03-model_mlp-moment_1-losscorcorr_8.45e-02-lossdis_1.67e-05-condvec_1"
    ]
    # metrics

    D = get_dataset("adult")
    discrete_columns = D.discrete_columns
    continuous_columns = [c for c in list(D.data_train) if c not in discrete_columns]
    print(discrete_columns)
    print(continuous_columns)

    for folder in eval_folders:
        print(f">> {folder}")
        df_score = compute_statistical_metrics(
            None,
            folder,
            discrete_columns,
            continuous_columns,
            mode="last",
        )

    # ML performance
    for folder in eval_folders:
        df_score_ml = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode=mode,
            task="single",
        )

    # ML augmentation performance
    for folder in eval_folders:
        df_score_augment = compute_ml_metrics_all_ml_methods(
            D,
            None,
            folder,
            mode=mode,
            task="augment",
        )

    print(df_score)
    print(df_score_ml)
    print(df_score_augment)

    a = 2


if __name__ == "__main__":
    # main()
    # test()
    score, dataset = init_score(
        # "test-higgs-small-ctgan-lv_2-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        # "higgs-small-ctgan-lv_2-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        # "test-higgs-small-ctgan-lv_2-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        # "adult-ctgan-lv_2-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        # "house_16h-ctgan-lv_0-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        "higgs_small-ctgan-lv_0-bs_1000-epochs_100-ed_64-dd_32-gd_64-glr_1.42e-05-moment_2-losscorcorr_3.55e-02-lossdis_1.10e-08-condvec_1",
        "abc",
    )

    print(score, dataset)
