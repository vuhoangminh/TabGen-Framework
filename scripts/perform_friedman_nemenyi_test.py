import os
import scipy.stats as stat
import numpy as np
import pandas as pd
import re
import scikit_posthocs as sp
import engine.utils.hyperopt_utils as hyperopt_utils
from engine.config import config


__all__ = [
    "multivariate_normal",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "F_score",
    "fleiss_kappa",
    "r2_score",
    "compute_ranks",
    "nemenyi_test",
]


def r2_score(y_true, y_pred):
    """R squared (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.
    Returns
    -------
    z : float
        The R^2 score.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from parsimony.utils.stats import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            return 0.0

    return 1 - numerator / denominator


def _critical_nemenyi_value(p_value, num_models):
    """Critical values for the Nemenyi test.
    Table obtained from: https://gist.github.com/garydoranjr/5016455
    """
    values = [  # p   0.01   0.05   0.10    Models
        [2.576, 1.960, 1.645],  # 2
        [2.913, 2.344, 2.052],  # 3
        [3.113, 2.569, 2.291],  # 4
        [3.255, 2.728, 2.460],  # 5
        [3.364, 2.850, 2.589],  # 6
        [3.452, 2.948, 2.693],  # 7
        [3.526, 3.031, 2.780],  # 8
        [3.590, 3.102, 2.855],  # 9
        [3.646, 3.164, 2.920],  # 10
        [3.696, 3.219, 2.978],  # 11
        [3.741, 3.268, 3.030],  # 12
        [3.781, 3.313, 3.077],  # 13
        [3.818, 3.354, 3.120],  # 14
        [3.853, 3.391, 3.159],  # 15
        [3.884, 3.426, 3.196],  # 16
        [3.914, 3.458, 3.230],  # 17
        [3.941, 3.489, 3.261],  # 18
        [3.967, 3.517, 3.291],  # 19
        [3.992, 3.544, 3.319],  # 20
        [4.015, 3.569, 3.346],  # 21
        [4.037, 3.593, 3.371],  # 22
        [4.057, 3.616, 3.394],  # 23
        [4.077, 3.637, 3.417],  # 24
        [4.096, 3.658, 3.439],  # 25
        [4.114, 3.678, 3.459],  # 26
        [4.132, 3.696, 3.479],  # 27
        [4.148, 3.714, 3.498],  # 28
        [4.164, 3.732, 3.516],  # 29
        [4.179, 3.749, 3.533],  # 30
        [4.194, 3.765, 3.550],  # 31
        [4.208, 3.780, 3.567],  # 32
        [4.222, 3.795, 3.582],  # 33
        [4.236, 3.810, 3.597],  # 34
        [4.249, 3.824, 3.612],  # 35
        [4.261, 3.837, 3.626],  # 36
        [4.273, 3.850, 3.640],  # 37
        [4.285, 3.863, 3.653],  # 38
        [4.296, 3.876, 3.666],  # 39
        [4.307, 3.888, 3.679],  # 40
        [4.318, 3.899, 3.691],  # 41
        [4.329, 3.911, 3.703],  # 42
        [4.339, 3.922, 3.714],  # 43
        [4.349, 3.933, 3.726],  # 44
        [4.359, 3.943, 3.737],  # 45
        [4.368, 3.954, 3.747],  # 46
        [4.378, 3.964, 3.758],  # 47
        [4.387, 3.973, 3.768],  # 48
        [4.395, 3.983, 3.778],  # 49
        [4.404, 3.992, 3.788],  # 50
    ]

    if num_models < 2 or num_models > 50:
        raise ValueError("num_models must be in [2, 50].")

    if p_value == 0.01:
        return values[num_models - 2][0]
    elif p_value == 0.05:
        return values[num_models - 2][1]
    elif p_value == 0.10:
        return values[num_models - 2][2]
    else:
        raise ValueError("p_value must be in {0.01, 0.05, 0.10}")


def compute_ranks(X, method="average"):
    """Assign ranks to data, dealing with ties appropriately.
    Uses scipy.stats.rankdata to compute the ranks of each row of the matrix X.
    Parameters
    ----------
    X : numpy array
        Computes the ranks of the rows of X.
    method : str
        The method used to assign ranks to tied elements. Must be one of
        "average", "min", "max", "dense" and "ordinal".
    Returns
    -------
    R : numpy array
        A matrix with the ranks computed from X. Has the same shape as X. Ranks
        begin at 1.
    """
    if method not in ["average", "min", "max", "dense", "ordinal"]:
        raise ValueError(
            'Method must be one of "average", "min", "max", ' '"dense" and "ordinal".'
        )

    n = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(n):
        r = stat.rankdata(X[i, :], method=method)
        R[i, :] = r

    return R


def nemenyi_test(
    X, p_value=0.05, return_ranks=True, return_critval=True, return_p_value=True
):
    """Performs the Nemenyi test for comparing a set of classifiers to each
    other.
    Parameters
    ----------
    X : numpy array of shape (num_datasets, num_models)
        The scores of the num_datasets datasets for each of the num_models
        models. X must have at least one row and between 2 and 50 columns.
    p_value : float
        The p-value of the test. Must be one of 0.01, 0.05 or 0.1. Default is
        p_value=0.05.
    return_ranks : bool
        Whether or not to return the computed ranks. Default is False, do not
        return the ranks.
    return_critval : bool
        Whether or not to return the computed critical value. Default is False,
        do not return the critical value.
    """
    num_datasets, num_models = X.shape
    R = compute_ranks(X)
    crit_val = _critical_nemenyi_value(p_value, num_models)
    CD = crit_val * np.sqrt(num_models * (num_models + 1) / (6.0 * num_datasets))

    sign = np.zeros((num_models, num_models))

    sign_bool = np.zeros((num_models, num_models))
    for j1 in range(num_models):
        for j2 in range(num_models):
            if np.abs(np.mean(R[:, j1] - R[:, j2])) > CD:
                sign[j1, j2] = 1
            else:
                sign[j1, j2] = 0
            if np.mean(R[:, j1] - R[:, j2]) > 0:
                sign_bool[j1, j2] = 1
            else:
                sign_bool[j1, j2] = -1

    # Perform Nemenyi post-hoc test
    data = X
    nemenyi_p_values = sp.posthoc_nemenyi_friedman(data)

    return sign, R, CD, sign_bool, nemenyi_p_values


# ==========================================================================================
# Added from here
# ==========================================================================================

LIST_DATASETS = [
    # small
    "abalone",  # done
    "wilt",  # done
    "churn2",  # done
    "diabetes_openml",  # done
    "buddy",  # done
    "gesture",  # done
    "insurance",  # done
    "king",  # done
    "cardio",  # done
    "california",  # done
    "house_16h",  # done
    "adult",  # done
    "house",  # done
    "news",  # done
    "diabetesbalanced",  # 4
    # large
    "diabetes",  # 9
    "mnist12",
    # super large
    "higgs-small",  # fixed name
    "miniboone",
    "credit",
]


LIST_METHOD_CONDVEC = ["ctgan", "copulagan", "dpcgans"]

DICT_MAPPING_METRICS = {  # higher/lower is better
    "kl_divergence_discrete": "lower",
    "kl_divergence_continuous": "lower",
    "chisquare_discrete": "lower",
    "kolmogorov_smirnov_continuous": "higher",
    "cramer_discrete": "higher",
    "pearson_continuous": "higher",
    "dwp_discrete": "lower",
    "dwp_continuous": "lower",
}


def perform_test(df, is_write=True, verbose=1, return_p_value=False, return_rank=False):
    df_data = df.iloc[:, :]
    X = df_data.values

    models = list(df.columns.values)
    models.append("total")

    p_value = 0.05  # 95%
    # p_value = 0.01  # 99%
    sign, R, CD, sign_bool, nemenyi_p_values = nemenyi_test(X, p_value=p_value)

    if return_p_value:
        x = sign_bool
        column_sums = x.sum(axis=1) + 1
    else:
        x = np.multiply(sign, sign_bool)
        column_sums = x.sum(axis=1)

    # Create a new column for the sum values
    arr_with_sum_column = np.column_stack((x, column_sums))

    df_result = pd.DataFrame(arr_with_sum_column, columns=models)

    index_list = list(df.columns.values)
    index_series = pd.Series(index_list)

    # Set the index using the Series
    df_result = df_result.set_index(index_series)
    df_result_new = df_result.copy()

    # Iterate over rows and columns
    i_row = 0
    for index, row in df_result.iterrows():
        i_col = 0
        for col in df_result.columns:
            if index == col:
                df_result.loc[index, col] = ""
                df_result_new.loc[index, col] = ""
            else:
                if col != "total":
                    if df_result.loc[index, col] == 0:
                        df_result.loc[index, col] = "0"
                        df_result_new.loc[index, col] = "0"
                    elif df_result.loc[index, col] > 0:
                        df_result.loc[index, col] = "+"
                        df_result_new.loc[index, col] = "+"
                    else:
                        df_result.loc[index, col] = "-"
                        df_result_new.loc[index, col] = "-"
                    if return_p_value:
                        df_result.loc[index, col] = (
                            df_result.loc[index, col]
                            + f"(p={float(nemenyi_p_values.values[i_row, i_col]):.3f})"
                        )
                        if nemenyi_p_values.values[i_row, i_col] <= 0.01:
                            df_result_new.loc[index, col] = (
                                df_result_new.loc[index, col] * 3
                            )
                        elif nemenyi_p_values.values[i_row, i_col] <= 0.05:
                            df_result_new.loc[index, col] = (
                                df_result_new.loc[index, col] * 2
                            )
            i_col += 1
        i_row += 1

    if return_rank:
        if return_p_value:
            return (
                df_result.drop(columns=df_result.columns[-1]),
                df_result_new.drop(columns=df_result.columns[-1]),
                R,
            )
        else:
            return df_result, R.mean(axis=0)
    else:
        if return_p_value:
            return (
                df_result.drop(columns=df_result.columns[-1]),
                df_result_new.drop(columns=df_result.columns[-1]),
            )
        else:
            return df_result


def get_group_datasets(group):
    if group == "regression":
        return ["house", "news"]
    elif group == "classification":
        return [
            "adult",
            "covertype",
            "credit",
            "diabetes",
            "diabetesbalanced",
            "intrusion",
            "mnist12",
            "mnist28",
        ]
    elif group == "all":
        return LIST_DATASETS
    elif group == "mnist":
        return ["mnist12", "mnist28"]
    elif group == "only_discrete":
        return ["diabetes", "diabetesbalanced", "mnist12", "mnist28"]
    elif group == "only_continuous":
        return ["credit"]
    elif group == "small":  # < 100 000 samples
        return [
            "adult",
            "diabetes",
            "mnist12",
            "mnist28",
            "house",
            "news",
        ]
    elif group == "big":
        return [
            "covertype",
            "credit",
            "diabetesbalanced",
            "intrusion",
        ]
    else:
        raise ValueError(f"{group} is not defined...")


def get_group_evaluation(evaluation):
    if evaluation == "statistics":
        return ""
    elif evaluation == "ml":
        return "_ml"
    elif evaluation == "ml_augment":
        return "_ml_augment"
    else:
        raise ValueError(f"{evaluation} is not defined...")


def get_group_method():
    return [
        "ctgan",
        "tvae",
        "copulagan",
        "dpcgans",
        "ctab",
    ]


def get_df(dataset, evaluation, mode, notebook_path=None):
    path = f"database/evaluation/{mode}/{dataset}{get_group_evaluation(evaluation)}.csv"

    if notebook_path is not None:
        path = os.path.join(notebook_path, path)

    df = pd.read_csv(path, index_col=0, header=0, sep="\t")

    if evaluation == "statistics":
        df = df.iloc[:, :-2]
    return df


def get_row_score(df, model, losscorr, lossdwp, condvec, moment=None):
    if moment is None:
        row = df.loc[
            (df["model"] == model)
            & (df["losscorr"] == losscorr)
            & (df["lossdwp"] == lossdwp)
            & (df["condvec"] == condvec)
        ]
    else:
        row = df.loc[
            (df["model"] == model)
            & (df["losscorr"] == losscorr)
            & (df["lossdwp"] == lossdwp)
            & (df["condvec"] == condvec)
            & (df["moment"] == moment)
        ]

    return row


def preprocess_metrics(row, evaluation):
    row = row.replace([np.inf], 10_000_000_000)
    row = row.replace([np.nan], 10_000_000_000)

    for metric in row.columns:
        if evaluation == "statistics":
            if DICT_MAPPING_METRICS[metric] == "lower":
                row[metric] = -row[metric]
        elif evaluation == "ml":  # lower is better
            row[metric] = -row[metric]
        elif (
            evaluation == "ml_augment"
        ):  # higher is better except for regression task (mae + mse)
            if "mae" in metric or "mse" in metric:
                row[metric] = -row[metric]  # regression task
    return row


def extract_score_list(row, evaluation):
    row = row.drop(
        columns=[
            "folder",
            "filename",
            "model",
            "losscorr",
            "lossdwp",
            "condvec",
        ]
    )

    row = preprocess_metrics(row, evaluation)

    return row.values.tolist()[0]


def print_latex(df_score):
    def extract_numbers(text):
        numbers = re.findall(r"\d+", text)
        return "".join(str(num) for num in numbers)

    # Iterate over rows and columns
    i_row = 0
    text = ""
    for index, row in df_score.iterrows():
        text += "{:<5s}".format(f"${extract_numbers(index)}$")
        for col in df_score.columns:
            if index == col:
                text += " & {:<5s}".format(f"~~~")
            else:
                text += " & {:<5s}".format(f"${df_score.loc[index, col]}$")
        text += " \\\\ \n"
    return print(text)


def get_bold_style(d, w):
    if d > 0.5:
        return f"\\hl{{{w}}}"
    elif d == 0.5:
        return f"\\zr{{{w}}}"
    else:
        return f"\\nt{{{w}}}"


def print_latex_methods(d):
    # Iterate over rows and columns
    i_row = 0
    text = ""
    for key in [
        "method",
        "statistics",
        "ml",
        "ml_augment",
        "comp",
        "empty",
        "statistics_wr",
        "ml_wr",
        "ml_augment_wr",
        "comp_wr",
    ]:
        if key == "method":
            text += "{:<20s}".format(f"\\h{{{config.DICT_MAPPING_MODEL[d[key]]}}}")
            text += "&"
        elif key == "dataset":
            text += "{:<20s}".format(f"{d[key]}")
            text += "&"
        elif key == "empty":
            text += " &".format(f"${d[key]}$")
        elif "_wr" in key:
            key_ = key + "se"
            s = get_bold_style(d[key], d[key_])
            text += " & {:<20s}".format(f"{s}")
        elif "_wrse" in key:
            pass
        else:
            if d[key] == "--":
                k = "-"
            elif d[key] == "++":
                k = "+"
            elif d[key] == "---":
                k = "--"
            elif d[key] == "+++":
                k = "++"
            else:
                k = "0"
            text += " & {:<5s}".format(f"${k}$")
    text += " \\\\"
    return print(text)


def print_latex_datasets(d):
    # Iterate over rows and columns
    i_row = 0
    text = ""
    for key in [
        "dataset",
        "statistics",
        "ml",
        "ml_augment",
        "comp",
        "empty",
        "statistics_wr",
        "ml_wr",
        "ml_augment_wr",
        "comp_wr",
    ]:
        if key == "dataset":
            s = config.DICT_STRING_DATASETS[d[key]]
            s = f"\\h{{{s}}}"
            text += "{:<20s}".format(s)
            text += "&"
        elif key == "empty":
            text += " &".format(f"${d[key]}$")
        elif "_wr" in key:
            key_ = key + "se"
            s = get_bold_style(d[key], d[key_])
            text += " & {:<20s}".format(f"{s}")
        elif "_wrse" in key:
            pass
        else:
            if d[key] == "--":
                k = "-"
            elif d[key] == "++":
                k = "+"
            elif d[key] == "---":
                k = "--"
            elif d[key] == "+++":
                k = "++"
            else:
                k = "0"
            text += " & {:<5s}".format(f"${k}$")
    text += " \\\\"
    return print(text)


def print_latex_losses(df_score):
    def extract_numbers(text):
        numbers = re.findall(r"\d+", text)
        return "".join(str(num) for num in numbers)

    # Iterate over rows and columns
    i_row = 0
    text = ""
    for index, row in df_score.iterrows():
        text += "{:<5s}".format(f"${extract_numbers(index)}$")
        for col in df_score.columns:
            if index in col:
                text += " & {:<5s}".format(f"~~~")
            else:
                text += " & {:<5s}".format(f"${df_score.loc[index, col]}$")
        text += " \\\\ \n"
    return print(text)


def bootstrap_standard_error(data, num_bootstrap_samples=1000):
    """
    Calculate the bootstrap estimate of standard error for a numpy array of 0s and 1s.

    Parameters:
    data (numpy array): Array containing 0s and 1s.
    num_bootstrap_samples (int): Number of bootstrap samples to generate. Default is 1000.

    Returns:
    float: The bootstrap estimate of the standard error.
    """
    # Store bootstrap sample means
    bootstrap_means = []

    # Perform bootstrap sampling
    for _ in range(num_bootstrap_samples):
        # Generate a bootstrap sample by randomly sampling with replacement from the data
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample and store it
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Calculate the standard error as the standard deviation of the bootstrap sample means
    standard_error = np.std(bootstrap_means)

    return standard_error


def write_df_score(
    datasets,
    evaluations=["statistics", "ml", "ml_augment"],
    models=None,
    condvecs=None,
    # dir="database/optimization",
    dir="database/20241112_optimization_mixed_loss_separate_loss_num_to_zero_done",
    verbose=True,
):
    def udpate_df(
        _df_score, _path_loss0, _path_loss2, dataset, model, condvec=1, model_type=""
    ):
        if os.path.exists(_path_loss0) and os.path.exists(_path_loss2):
            I0 = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
                _path_loss0, is_print=False
            )
            I0.update_trials_losses(evaluations=["statistics", "ml", "ml_augment"])
            I2 = hyperopt_utils.IncrementalObjectiveOptimizationGenerativeModel(
                _path_loss2, is_print=False
            )
            I2.update_trials_losses(evaluations=["statistics", "ml", "ml_augment"])

            for evaluation in evaluations:
                r0 = I0.get_row_fmin_evaluation(0, evaluation)
                r2 = I2.get_row_fmin_evaluation(2, evaluation)
                df_temp = pd.merge(r0, r2, on="metric", how="inner")
                df_temp.insert(0, "evaluation", evaluation)
                df_temp.insert(0, "model_type", model_type)
                df_temp.insert(0, "condvec", condvec)
                df_temp.insert(0, "model", model)
                df_temp.insert(0, "dataset", dataset)

                if _df_score is None:
                    _df_score = (
                        df_temp.copy()
                    )  # Use copy to avoid referencing the same object
                else:
                    _df_score = pd.concat(
                        [_df_score, df_temp], axis=0, ignore_index=True
                    )

        return _df_score

    path_df_score = f"{dir}/df_score.csv"
    if os.path.exists(path_df_score):
        df_score = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
        return df_score
    else:
        df_score = None

    for dataset in datasets:
        print(f"    >> dataset: {dataset}") if verbose else None
        for condvec in condvecs:
            print(f"        >> condvec: {condvec}") if verbose else None
            for model in models:
                print(f"            >> model: {model}") if verbose else None
                if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                    (
                        print(
                            f"            >> ignore model: {model} and condvec: {condvec}"
                        )
                        if verbose
                        else None
                    )
                    continue

                if model == "tabddpm":
                    for model_type in ["mlp", "resnet"]:
                        # for model_type in ["mlp"]:
                        (
                            print(f"                >> model_type: {model_type}")
                            if verbose
                            else None
                        )
                        path_loss0 = f"{dir}/{dataset}_{model}_loss_version-0_model-{model_type}.hyperopt"
                        path_loss2 = f"{dir}/{dataset}_{model}_loss_version-2_model-{model_type}.hyperopt"

                        (
                            print(f"                    >> {path_loss0}")
                            if verbose
                            else None
                        )
                        (
                            print(f"                    >> {path_loss2}")
                            if verbose
                            else None
                        )

                        df_score = udpate_df(
                            df_score,
                            path_loss0,
                            path_loss2,
                            dataset,
                            model,
                            condvec,
                            model_type=model_type,
                        )

                else:
                    if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                        (
                            print(
                                f"            >> ignore model: {model} and condvec: {condvec}"
                            )
                            if verbose
                            else None
                        )
                        continue

                    path_loss0 = (
                        f"{dir}/{dataset}_{model}_loss_version-0-{condvec}.hyperopt"
                    )
                    path_loss2 = (
                        f"{dir}/{dataset}_{model}_loss_version-2-{condvec}.hyperopt"
                    )

                    print(f"                >> {path_loss0}") if verbose else None
                    print(f"                >> {path_loss2}") if verbose else None

                    df_score = udpate_df(
                        df_score,
                        path_loss0,
                        path_loss2,
                        dataset,
                        model,
                        condvec,
                        model_type="",
                    )

    df_score.to_csv(path_df_score, sep="\t", encoding="utf-8")
    return df_score


def get_df_score_comparison(
    datasets,
    evaluations,
    models=None,
    condvecs=None,
    model_types=None,
    # dir="database/optimization",
    dir="database/20241112_optimization_mixed_loss_separate_loss_num_to_zero_done",
    verbose=False,
    is_read_existing=True,
):

    def get_row(_df_score_full, dataset, model, condvec, evaluation, model_type=None):
        """Checks if a single row in the DataFrame has the specified values in columns 'dataset', 'model', 'condvec', and if the 'metric' column contains a substring of the specified evaluation.

        Args:
            _df_score_full: The pandas DataFrame.
            dataset: The value for the 'dataset' column.
            model: The value for the 'model' column.
            condvec: The value for the 'condvec' column.
            evaluation: The value for the 'evaluation' column.
            model_type: The value for the 'model_type' column (optional).

        Returns:
            The matching row in the DataFrame, or None if no match is found.
        """

        if model_type is None:
            return _df_score_full[
                (_df_score_full["dataset"] == dataset)
                & (_df_score_full["model"] == model)
                & (_df_score_full["condvec"] == condvec)
                & (_df_score_full["evaluation"] == evaluation)
            ]
        else:
            return _df_score_full[
                (_df_score_full["dataset"] == dataset)
                & (_df_score_full["model"] == model)
                & (_df_score_full["condvec"] == condvec)
                & (_df_score_full["model_type"] == model_type)
                & (_df_score_full["evaluation"] == evaluation)
            ]

    path_df_score = f"{dir}/df_score.csv"
    if os.path.exists(path_df_score):
        df_score_full = pd.read_csv(path_df_score, sep="\t", header=0, index_col=0)
    else:
        raise FileExistsError

    df_score = None
    for evaluation in evaluations:
        for dataset in datasets:
            print(f"    >> dataset: {dataset}") if verbose else None
            for condvec in condvecs:
                print(f"        >> condvec: {condvec}") if verbose else None
                for model in models:
                    print(f"            >> model: {model}") if verbose else None
                    if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                        (
                            print(
                                f"            >> ignore model: {model} and condvec: {condvec}"
                            )
                            if verbose
                            else None
                        )
                        continue

                    if model == "tabddpm":
                        for model_type in model_types:
                            (
                                print(f"                >> model_type: {model_type}")
                                if verbose
                                else None
                            )
                            path_loss0 = f"{dir}/{dataset}_{model}_loss_version-0_model-{model_type}.hyperopt"
                            path_loss2 = f"{dir}/{dataset}_{model}_loss_version-2_model-{model_type}.hyperopt"

                            (
                                print(f"                    >> {path_loss0}")
                                if verbose
                                else None
                            )
                            (
                                print(f"                    >> {path_loss2}")
                                if verbose
                                else None
                            )

                            df_temp = get_row(
                                df_score_full,
                                dataset,
                                model,
                                condvec,
                                evaluation,
                                model_type,
                            )
                            if df_score is None:
                                df_score = (
                                    df_temp.copy()
                                )  # Use copy to avoid referencing the same object
                            else:
                                df_score = pd.concat(
                                    [df_score, df_temp], axis=0, ignore_index=True
                                )

                    else:
                        if condvec == 0 and model in ["ctab", "tvae", "tabddpm"]:
                            (
                                print(
                                    f"            >> ignore model: {model} and condvec: {condvec}"
                                )
                                if verbose
                                else None
                            )
                            continue

                        path_loss0 = (
                            f"{dir}/{dataset}_{model}_loss_version-0-{condvec}.hyperopt"
                        )
                        path_loss2 = (
                            f"{dir}/{dataset}_{model}_loss_version-2-{condvec}.hyperopt"
                        )

                        print(f"                >> {path_loss0}") if verbose else None
                        print(f"                >> {path_loss2}") if verbose else None

                        df_temp = get_row(
                            df_score_full,
                            dataset,
                            model,
                            condvec,
                            evaluation,
                        )
                        if df_score is None:
                            df_score = (
                                df_temp.copy()
                            )  # Use copy to avoid referencing the same object
                        else:
                            df_score = pd.concat(
                                [df_score, df_temp], axis=0, ignore_index=True
                            )

    return df_score


# ==========================================================================================
# Added end here
# ==========================================================================================


def test():
    datasets = get_group_datasets("all")
    # evaluations = ["statistics", "ml", "ml_augment"]
    evaluations = ["statistics"]
    evaluations = ["ml"]

    df_score = get_df_score_comparison(datasets, evaluations, mode="optimal")

    print(df_score)

    # Perform Friedman Test
    friedman_result = stat.friedmanchisquare(
        df_score.iloc[:, 0],
        df_score.iloc[:, 1],
        df_score.iloc[:, 2],
        df_score.iloc[:, 3],
    )

    print(friedman_result)

    # Perform Nemenyi post-hoc test
    data = np.array(
        [
            df_score.iloc[:, 0],
            df_score.iloc[:, 1],
            df_score.iloc[:, 2],
            df_score.iloc[:, 3],
        ]
    )
    nemenyi_p_values = sp.posthoc_nemenyi_friedman(data.T)

    print(nemenyi_p_values)


def main(is_remove_same_value=False):

    list_datasets = [
        "abalone",  # done
        "house",  # done
        "news",  # done
        "churn2",  # done
        "buddy",  # done
        "insurance",  # done
        "king",  # done
        "adult",  # done
        "cardio",  # done
        "wilt",  # done
        "diabetes_openml",  # done
        "gesture",  # done
        "california",  # done
        "house_16h",  # done
        "diabetesbalanced",  # done
        "diabetes",  # done
        "mnist12",  # done
        "credit",  # done
        "higgs-small",  # done
        "miniboone",  # done
    ]

    list_models = [
        "ctgan",
        "copulagan",
        "dpcgans",
        "ctab",
        "tvae",
        "tabddpm",
    ]

    for models in [
        list_models,
    ]:
        for condvecs in [
            [0, 1],
        ]:

            for evaluations in [
                ["statistics"],
                ["ml"],
                ["ml_augment"],
                ["statistics", "ml", "ml_augment"],
            ]:
                print(f">> processing {evaluations}")

                df_score = get_df_score_comparison(
                    list_datasets,
                    evaluations,
                    models=models,
                    condvecs=condvecs,
                    model_types=["mlp", "resnet"],
                )

                df_score = df_score.iloc[:, -2:]

                if is_remove_same_value:
                    df_score = df_score[
                        ~df_score.apply(lambda row: all(row == row[0]), axis=1)
                    ]

                result, result2, r = perform_test(
                    df_score, verbose=0, return_p_value=True, return_rank=True
                )
                # print(tabulate(result, headers="keys", tablefmt="fancy_grid"))
                # print(tabulate(result2, headers="keys", tablefmt="fancy_grid"))

                print()
                print()
                print()
                print(evaluations, models, condvecs)
                print()
                x = r[:, 1]
                rank = np.round(np.mean(x) - 1, 3)
                # se = np.std(x, ddof=1) / np.sqrt(np.size(x))
                # se = np.round(se, 3)

                se = bootstrap_standard_error(x)
                se = np.round(se, 3)
                print(f"{rank} ({se})")
                print()
                print()
                print_latex(result2)
                print()
                print()
                print()

                a = 2


def main_compare_datasets():
    list_datasets = get_group_datasets("all")
    list_datasets.sort()

    models = [
        "ctgan",
        "copulagan",
        "dpcgans",
        "ctab",
        "tvae",
        "tabddpm",
    ]

    for datasets in list_datasets:

        for condvecs in [
            [0, 1],
        ]:
            for model_types in [
                ["mlp", "resnet"],
            ]:

                d = {}

                d["dataset"] = datasets

                try:
                    for evaluations in [
                        ["statistics"],
                        ["ml"],
                        ["ml_augment"],
                        ["statistics", "ml", "ml_augment"],
                    ]:

                        df_score = get_df_score_comparison(
                            [datasets],
                            evaluations,
                            models=models,
                            condvecs=condvecs,
                            model_types=model_types,
                        )

                        df_score = df_score.iloc[:, -2:]

                        result, result2, r = perform_test(
                            df_score, verbose=0, return_p_value=True, return_rank=True
                        )

                        x = r[:, 1]
                        rank = np.round(np.mean(x) - 1, 3)
                        # se = np.std(x, ddof=1) / np.sqrt(np.size(x))
                        # se = np.round(se, 3)

                        se = bootstrap_standard_error(x)
                        se = np.round(se, 3)

                        if len(evaluations) > 1:
                            d[f"comp"] = result2["0"]["2"]
                            d[f"comp_wr"] = rank
                            d[f"comp_wrse"] = "{:.3f} ({:.3f})".format(rank, se)
                        else:
                            d[f"{evaluations[0]}"] = result2["0"]["2"]
                            d[f"{evaluations[0]}_wr"] = rank
                            d[f"{evaluations[0]}_wrse"] = "{:.3f} ({:.3f})".format(
                                rank, se
                            )

                    d["empty"] = ""

                    print_latex_datasets(d)

                except:
                    pass

    a = 2


def main_compare_methods():
    datasets = get_group_datasets("all")

    for models in [
        ["ctgan"],
        ["tvae"],
        ["copulagan"],
        ["dpcgans"],
        ["ctab"],
        ["tabddpm"],
        # ["tabsyn"],
    ]:

        for condvecs in [
            [1],
            [0],
        ]:

            for model_types in [
                ["mlp"],
                ["resnet"],
            ]:
                if models != ["tabddpm"] and model_types == ["mlp"]:
                    continue

                d = {}

                if condvecs == [1]:
                    d["method"] = models[0]
                else:
                    d["method"] = models[0] + "*"

                if models == ["tabddpm"]:
                    d["method"] += f"-{model_types[0]}"

                try:
                    for evaluations in [
                        ["statistics"],
                        ["ml"],
                        ["ml_augment"],
                        ["statistics", "ml", "ml_augment"],
                    ]:

                        df_score = get_df_score_comparison(
                            datasets,
                            evaluations,
                            models=models,
                            condvecs=condvecs,
                            model_types=model_types,
                        )

                        df_score = df_score.iloc[:, -2:]

                        result, result2, r = perform_test(
                            df_score, verbose=0, return_p_value=True, return_rank=True
                        )

                        x = r[:, 1]
                        rank = np.round(np.mean(x) - 1, 3)
                        # se = np.std(x, ddof=1) / np.sqrt(np.size(x))
                        # se = np.round(se, 3)

                        se = bootstrap_standard_error(x)
                        se = np.round(se, 3)

                        if len(evaluations) > 1:
                            d[f"comp"] = result2["0"]["2"]
                            d[f"comp_wr"] = rank
                            d[f"comp_wrse"] = "{:.3f} ({:.3f})".format(rank, se)
                        else:
                            d[f"{evaluations[0]}"] = result2["0"]["2"]
                            d[f"{evaluations[0]}_wr"] = rank
                            d[f"{evaluations[0]}_wrse"] = "{:.3f} ({:.3f})".format(
                                rank, se
                            )

                    d["empty"] = ""

                    print_latex_methods(d)

                except:
                    pass

    a = 2


def main_compare_losses():
    datasets = get_group_datasets("all")

    df_score_datasets = None
    for models in [
        [
            "ctgan",
            "tvae",
            "copulagan",
            "dpcgans",
            "ctab",
        ],
    ]:
        for condvecs in [
            [0, 1],
        ]:
            for evaluations in [
                ["statistics"],
                ["ml"],
                ["ml_augment"],
                ["statistics", "ml", "ml_augment"],
            ]:
                r = None
                for mode in ["optimal", "last"]:

                    df_score = get_df_score_comparison(
                        datasets,
                        evaluations,
                        mode=mode,
                        models=models,
                        condvecs=condvecs,
                    )

                    result, result2 = perform_test(
                        df_score, verbose=0, return_p_value=True
                    )

                    result2 = result2.add_prefix(f"{mode}_")

                    if r is None:
                        r = result2
                    else:
                        r = pd.concat([r, result2], axis=1)

                print_latex_losses(r)

                a = 2


def main_compare_each():
    list_datasets = get_group_datasets("all")

    df_score_datasets = None

    for datasets in list_datasets:
        print()
        print()
        print(datasets)
        print()
        print()
        for models in [
            ["ctgan"],
            ["tvae"],
            ["copulagan"],
            ["dpcgans"],
            ["ctab"],
        ]:
            for condvecs in [
                [1],
                [0],
                # [0, 1],
            ]:
                try:
                    d = {}
                    d["method"] = f"{models[0]}_{condvecs[0]}"
                    for mode in ["optimal", "last"]:
                        for evaluations in [
                            ["statistics"],
                            ["ml"],
                            ["ml_augment"],
                            ["statistics", "ml", "ml_augment"],
                        ]:

                            df_score = get_df_score_comparison(
                                [datasets],
                                evaluations,
                                mode=mode,
                                models=models,
                                condvecs=condvecs,
                            )

                            result, result2 = perform_test(
                                df_score, verbose=0, return_p_value=True
                            )

                            a = 2

                            if evaluations == ["statistics", "ml", "ml_augment"]:
                                d[f"{mode}_comp."] = result2.loc[
                                    "corr-1_dwp-1", "corr-0_dwp-0"
                                ]
                            else:
                                d[f"{mode}_{evaluations[0]}"] = result2.loc[
                                    "corr-1_dwp-1", "corr-0_dwp-0"
                                ]

                        d["empty"] = ""

                    print_latex_methods(d)

                except:
                    pass

        a = 2


if __name__ == "__main__":

    write_df_score(
        datasets=[
            "abalone",  # done
            "house",  # done
            "news",  # done
            "churn2",  # done
            "buddy",  # done
            "insurance",  # done
            "king",  # done
            "adult",  # done
            "cardio",  # done
            "wilt",  # done
            "diabetes_openml",  # done
            "gesture",  # done
            "california",  # done
            "house_16h",  # done
            "diabetesbalanced",  # done
            "diabetes",  # done
            "mnist12",  # done
            "credit",  # done
            "higgs-small",  # done
            "miniboone",  # done
        ],
        models=[
            "ctgan",
            "copulagan",
            "dpcgans",
            "ctab",
            "tvae",
            "tabddpm",
            "tabsyn",
        ],
        condvecs=[0, 1],
    )

    # test()
    main()
    main_compare_methods()
    # main_compare_datasets()
    # main_compare_losses()
    # main_compare_each()
