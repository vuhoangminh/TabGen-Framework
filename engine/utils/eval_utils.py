import os
import math
import numpy as np
import pandas as pd
import itertools
from rich import print

from multiprocessing import Pool
from itertools import repeat

import engine.config as config

from scipy.stats import norm
from scipy.stats import entropy  # KL divergence
from scipy.stats import chisquare  # Pearson's chi-squared test
from scipy.stats import kstest  # Kolmogorov–Smirnov test
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr


# ===========================================================================
# Misc
# ===========================================================================
def is_sample_have_all_classes(df, df_fake):
    is_satisfied = True
    n_fail = 0
    for col in list(df):
        print(f"{col}: {df[col].nunique()}, {df_fake[col].nunique()}")
        if df[col].nunique() != df_fake[col].nunique():
            is_satisfied = False
            n_fail += 1
    print(f"-->> failed: {n_fail} / {len(list(df))}")
    return is_satisfied


# ===========================================================================
# Old KL divergence functions
# ===========================================================================
def _legacy_clip_array(a, b):
    """
    Clip the values of array b to the range of array a.

    Args:
        a (numpy.ndarray): The reference array.
        b (numpy.ndarray): The array to clip.

    Returns:
        numpy.ndarray: The clipped array.
    """
    a_min = a.min()
    a_max = a.max()
    return np.clip(b, a_min, a_max)


def _legacy_compute_kl_divergence(df, df_fake, continuous_columns=[]):
    d = 0
    count = 0
    for col in list(df.columns):
        if col in continuous_columns:
            df_array = df[col].to_numpy()
            df_fake_array = df_fake[col].to_numpy()

            df_fake_array = _legacy_clip_array(df_array, df_fake_array)

            # Reshape the arrays to have the same shape
            max_len = max(len(df_array), len(df_fake_array))
            df_array = np.pad(df_array, (0, max_len - len(df_array)))
            df_fake_array = np.pad(df_fake_array, (0, max_len - len(df_fake_array)))

            s = entropy(df_array, df_fake_array)

            if not np.isinf(s):
                d += s
                count += 1

    if count > 0:
        d /= count

    return d


# ===========================================================================
# Dimensional wise probability (DWP) functions
# ===========================================================================
def compute_distance_point_to_line(p3, p1=[0, 0], p2=[1, 1]):
    p3 = np.asarray(p3, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return d


def compute_dwp(
    df,
    df_fake,
    discrete_columns=[],
    is_included_discrete=True,
    is_included_continuous=False,
):
    x, y = [], []
    d = 0
    count = 0
    for col in list(df.columns):
        # discrete columns
        if col in discrete_columns:
            if is_included_discrete:
                uniques = list(df[col].unique())
                df_unique_counts = df[col].value_counts()
                df_fake_unique_counts = df_fake[col].value_counts()
                for u in uniques:
                    try:
                        a = df_unique_counts[u] / len(df)
                    except:
                        a = 0
                        # print(col, u)
                    try:
                        b = df_fake_unique_counts[u] / len(df_fake)
                    except:
                        b = 0
                        # print(col, u)
                    x.append(a)
                    y.append(b)

                    d += compute_distance_point_to_line([a, b])
                    count += 1

        # continuous columns
        else:
            if is_included_continuous:
                a = df[col].mean()
                b = df_fake[col].mean()
                x.append(a)
                y.append(b)

                d += compute_distance_point_to_line([a, b])
                count += 1

    if count > 0:
        d /= count

    return d, x, y


# ===========================================================================
# Correlation functions
# ===========================================================================
def compute_diff_correlation(df, df_fake):
    coef1 = df.select_dtypes("number").corr().fillna(0)
    coef2 = df_fake.select_dtypes("number").corr().fillna(0)
    d = np.mean(np.absolute(coef1 - coef2).values.reshape(-1))

    if math.isnan(d):
        df_copy = df.astype(float)
        df_fake_copy = df_fake.astype(float)
        coef1 = df_copy.select_dtypes("number").corr().fillna(0)
        coef2 = df_fake_copy.select_dtypes("number").corr().fillna(0)
        d = np.mean(np.absolute(coef1 - coef2).values.reshape(-1))

    return d


# ===========================================================================
# KL divergence functions
# ===========================================================================
def _legacy_get_bins_historgram(x):
    # Compute the Freedman-Diaconis bin width
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    bin_width = 2 * iqr * len(x) ** (-1 / 3)

    # Compute the number of bins using the bin width
    num_bins = int(np.ceil((x.max() - x.min()) / bin_width))

    return num_bins


def _legacy_get_norm_historgram(x, num_bins):
    x_hist, _ = np.histogram(x, bins=num_bins)
    x_hist_norm = x_hist / np.sum(x_hist)
    return x_hist_norm


def plot_hist(x, y, num_bins, col):
    import matplotlib.pyplot as plt

    # Plot the histograms
    plt.hist(x, bins=num_bins, alpha=0.5, label="Data 1")
    plt.hist(y, bins=num_bins, alpha=0.5, label="Data 2")
    plt.legend(loc="upper right")
    plt.title(col)
    plt.show()


def _legacy_get_value_counts_intersection(df1, df2, col):
    # Get the unique values of column 'X' in both DataFrames
    unique_values = pd.Series(list(set(df1[col]).union(set(df2[col]))))

    # Get the value counts of column 'X' in both DataFrames and fill with 0
    vc1 = df1[col].value_counts().reindex(unique_values, fill_value=0).sort_index()
    vc2 = df2[col].value_counts().reindex(unique_values, fill_value=0).sort_index()

    return vc1, vc2


def get_value_counts_intersection_categorical_variable(df1, df2, col):
    # Get the unique values of column 'X' in both DataFrames
    unique_values = pd.Series(list(set(df1[col]).intersection(set(df2[col]))))

    # Filter the rows of both DataFrames based on the unique values
    df1_filtered = df1[df1[col].isin(unique_values)]
    df2_filtered = df2[df2[col].isin(unique_values)]

    # Get the value counts of column 'X' in both DataFrames
    vc1 = df1_filtered.value_counts().sort_index()
    vc2 = df2_filtered.value_counts().sort_index()

    return vc1, vc2


def get_value_counts(df_column, values_to_count):
    """
    Get the value counts for each value in a list within a Pandas DataFrame column.
    If a value is not present in the column, return 0 for that value.

    Args:
        df_column (pd.Series): A Pandas Series (column) containing data.
        values_to_count (list): List of values to count.

    Returns:
        pd.Series: A Series with value counts for each value in the list.
    """
    value_counts = df_column[df_column.isin(values_to_count)].value_counts()

    # Initialize missing values with 0
    for value in values_to_count:
        if value not in value_counts.index:
            value_counts[value] = 0

    # Sort the result by index
    value_counts.sort_index(inplace=True)

    return value_counts


def get_value_counts_union_categorical_variable(df1, df2, col):
    # Get the unique values of column 'X' in both DataFrames
    values_to_count = list(set(df1[col]).union(set(df2[col])))

    # Get the value counts of column 'X' in both DataFrames
    vc1 = get_value_counts(df1[col], values_to_count)
    vc2 = get_value_counts(df2[col], values_to_count)

    return vc1, vc2


def estimate_distribution_continuous_variable(df1, df2, col):
    def kl_divergence(mu1, sigma1, mu2, sigma2):
        return (
            np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
            - 0.5
        )

    mu1, sigma1 = norm.fit(df1[col])
    mu2, sigma2 = norm.fit(df2[col])

    return kl_divergence(mu1, sigma1, mu2, sigma2)


def compute_kl_divergence(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str],
    normalize: bool = False,
    handle_missing: bool = True,
    is_continuous: bool = False,
) -> float:
    """Calculate the KL divergence between the given distributions.

    Args:
        df1 (pandas.DataFrame): original df
        df2 (pandas.DataFrame): synthetic df
        cols (list[str]): list of either categorical or continuous cols
        normalize (bool): Whether to normalize the data before calculating KL divergence (for continuous variables).
        handle_missing (bool): Whether to handle missing values. Defaults to True.

    Returns:
        result (float): The calculated KL divergence.
    """

    if handle_missing:
        # Handle missing values in both dataframes
        df1_filtered = df1[cols].dropna()
        df2_filtered = df2[cols].dropna()
    else:
        # Don't handle missing values
        df1_filtered = df1[cols]
        df2_filtered = df2[cols]

    # Normalize continuous columns in the filtered dataframes
    if normalize:
        df1_filtered = (df1_filtered - df1_filtered.mean()) / df1_filtered.std()
        df2_filtered = (df2_filtered - df2_filtered.mean()) / df2_filtered.std()

    result = 0
    for col in cols:
        if is_continuous:
            e = estimate_distribution_continuous_variable(
                df1_filtered, df2_filtered, col
            )
            # print(col, e)
        else:
            x, y = get_value_counts_union_categorical_variable(
                df1_filtered[[col]].copy(),
                df2_filtered[[col]].copy(),
                col,
            )

            x_hist = x.to_numpy()
            y_hist = y.to_numpy()

            x_hist_norm = x_hist / np.sum(x_hist)
            y_hist_norm = y_hist / np.sum(y_hist)

            e = entropy(x_hist_norm, y_hist_norm)
            # print(col, e)

        result += e

    if len(cols) == 0:
        return 0
    else:
        return result / len(cols)


# ===========================================================================
# Chi-squared test functions
# ===========================================================================
def compute_chisquare_test(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str],
    handle_missing: bool = True,
) -> tuple[float]:
    """Calculate a one-way chi-square test.

    Args:
        df1 (pd.DataFrame): original df
        df2 (pd.DataFrame): synthetic df
        cols (list[str]): list of categorical cols
        handle_missing (bool, optional): Whether to handle missing values. Defaults to True.

    Returns:
        statistic: The chi-squared test statistic. The value is a float if axis is None or f_obs and f_exp are 1-D.
        p_value: The p-value of the test. The value is a float if ddof and the return value chisq are scalars.

    """
    if len(cols) == 0:
        return 0

    if handle_missing:
        # Handle missing values by dropping rows with NaN
        df1_filtered = df1[cols].dropna()
        df2_filtered = df2[cols].dropna()

    else:
        # Don't handle missing values
        df1_filtered = df1[cols]
        df2_filtered = df2[cols]

    result = 0
    for col in cols:
        # x, y = get_value_counts_intersection_categorical_variable(
        #     df1_filtered[[col]].copy(),
        #     df2_filtered[[col]].copy(),
        #     col,
        # )

        x, y = get_value_counts_union_categorical_variable(
            df1_filtered[[col]].copy(),
            df2_filtered[[col]].copy(),
            col,
        )

        x_hist = x.to_numpy()
        y_hist = y.to_numpy()

        x_hist_norm = x_hist / np.sum(x_hist)
        y_hist_norm = y_hist / np.sum(y_hist)

        statistic, p_value = chisquare(x_hist_norm, y_hist_norm)

        # print(col, statistic, p_value)

        result += statistic

    return result / len(cols)


# ===========================================================================
# Kolmogorov–Smirnov test functions
# ===========================================================================
def compute_kolmogorov_smirnov_test(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str],
    handle_missing=True,
) -> tuple[float]:
    """Calculate the Kolmogorov-Smirnov test statistic and p-value.

    Args:
        df1 (pd.DataFrame): original df
        df2 (pd.DataFrame): synthetic df
        cols (list[str]): list of continuous cols
        two_tailed (bool, optional): Whether to calculate a two-tailed p-value. Defaults to False.
        handle_missing (bool, optional): Whether to handle missing values. Defaults to True.

    Returns:
        statistic: KS test statistic (D), the maximum of D+ and D-.
        p_value: One-tailed or two-tailed p-value.

    """

    if handle_missing:
        # Handle missing values by removing rows with NaN
        df1_filtered = df1[cols].dropna()
        df2_filtered = df2[cols].dropna()

    elif not handle_missing:
        # Don't handle missing values
        df1_filtered = df1[cols]
        df2_filtered = df2[cols]

    # Calculate the test statistic
    result = 0
    for col in cols:
        x = df1_filtered[col].copy()
        y = df2_filtered[col].copy()

        statistic, p_value = kstest(x, y)
        # print(col, statistic, p_value)

        result += statistic

    if len(cols) == 0:
        return 0
    else:
        return result / len(cols)


# ===========================================================================
# Cramer's V functions
# ===========================================================================
def compute_cramer(df, col1, col2):
    import researchpy

    crosstab, res = researchpy.crosstab(df[col1], df[col2], test="chi-square")
    s = res.iloc[2]["results"]
    return s


def compute_cramer_pair(index, pairs, d1, d2):
    print(f"{index} / {len(pairs)}")
    pair = pairs[index]
    c1 = compute_cramer(d1, pair[0], pair[1])
    c2 = compute_cramer(d2, pair[0], pair[1])
    return c1, c2, pair


def compute_cramers_v_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str],
    handle_missing: bool = True,
) -> float:
    """Calculate Cramer's V statistic and optionally chi-square p-value.

    Args:
        df1 (pd.DataFrame): original df
        df2 (pd.DataFrame): synthetic df
        cols (list[str]): list of categorical cols
        handle_missing (bool, optional): Whether to handle missing values. Defaults to True.
        p_value (bool, optional): Whether to calculate chi-square p-value. Defaults to False.

    Returns:
        cramers_v (float): Cramer's V statistic.
        p_value (float, optional): Chi-square p-value. Only returned if `p_value` is True.

    """

    if handle_missing:
        # Handle missing values by removing rows with NaN
        df1_filtered = df1[cols].dropna()
        df2_filtered = df2[cols].dropna()

    elif not handle_missing:
        # Don't handle missing values
        df1_filtered = df1[cols]
        df2_filtered = df2[cols]

    result = 0
    count = 0
    note = ""

    col_pairs = list(itertools.combinations(cols, 2))

    # for pair in col_pairs:
    #     # print(pair)
    #     c1 = compute_cramer(df1_filtered, pair[0], pair[1])
    #     c2 = compute_cramer(df2_filtered, pair[0], pair[1])

    #     if np.isnan(c1) or np.isnan(c2):  # ignore imbalanced column
    #         print(f"ignore nan {pair}")
    #         note = note + f"{pair}, "
    #     else:
    #         count += 1
    #         result += abs(c1 - c2)

    pool = Pool(16)
    l = pool.starmap(
        compute_cramer_pair,
        zip(
            range(len(col_pairs)),
            repeat(col_pairs),
            repeat(df1_filtered),
            repeat(df2_filtered),
        ),
    )
    pool.close()

    for i in range(len(col_pairs)):
        c1, c2, pair = l[i]

        if np.isnan(c1) or np.isnan(c2):  # ignore imbalanced column
            print(f"ignore nan {pair}")
            note = note + f"{pair}, "
        else:
            count += 1
            result += abs(c1 - c2)

    if count > 0:
        return result / count, note
    else:
        return 0, note


# ===========================================================================
# Pearson correlation functions
# ===========================================================================
def compute_pearson_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cols: list[str],
    handle_missing: bool = True,
) -> float:
    """Calculate Pearson correlation.

    Args:
        df1 (pd.DataFrame): original df
        df2 (pd.DataFrame): synthetic df
        cols (list[str]): list of continuous cols
        handle_missing (bool, optional): Whether to handle missing values. Defaults to True.

    Returns:
        pearson (float): Pearson correlation.

    """

    def compute_pearson(col1, col2):
        x = np.squeeze(col1.to_numpy())
        y = np.squeeze(col2.to_numpy())

        p = pearsonr(x, y)[0]
        return p

    if handle_missing:
        # Handle missing values by removing rows with NaN
        df1_filtered = df1[cols].dropna()
        df2_filtered = df2[cols].dropna()

    elif not handle_missing:
        # Don't handle missing values
        df1_filtered = df1[cols]
        df2_filtered = df2[cols]

    result = 0
    count = 0
    note = ""

    col_pairs = list(itertools.combinations(cols, 2))

    # print(col_pairs)

    for pair in col_pairs:
        # print(pair)
        col1_df1 = df1_filtered[[pair[0]]].copy()
        col2_df1 = df1_filtered[[pair[1]]].copy()

        col1_df2 = df2_filtered[[pair[0]]].copy()
        col2_df2 = df2_filtered[[pair[1]]].copy()

        c1 = compute_pearson(col1_df1, col2_df1)
        c2 = compute_pearson(col1_df2, col2_df2)

        if np.isnan(c1) or np.isnan(c2):  # ignore imbalanced column
            print(f"ignore nan {pair}")
            note = note + f"{pair}, "
        else:
            count += 1
            result += abs(c1 - c2)

    if count > 0:
        return result / count, note
    else:
        return 0, note


# ===========================================================================
# main
# ===========================================================================
def main_kl(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns):
    kl_cat = compute_kl_divergence(
        df, df_fake_old, cols=discrete_columns, normalize=False, handle_missing=False
    )
    print(kl_cat)

    kl_cat = compute_kl_divergence(
        df, df_fake_new, cols=discrete_columns, normalize=False, handle_missing=False
    )
    print(kl_cat)

    kl_con = compute_kl_divergence(
        df,
        df_fake_old,
        cols=continuous_columns,
        normalize=False,
        handle_missing=False,
        is_continuous=True,
    )
    print(kl_con)

    kl_con = compute_kl_divergence(
        df,
        df_fake_new,
        cols=continuous_columns,
        normalize=False,
        handle_missing=False,
        is_continuous=True,
    )
    print(kl_con)


def main_cs(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns):
    cs_cat = compute_chisquare_test(
        df, df_fake_old, cols=discrete_columns, handle_missing=False
    )
    print(cs_cat)

    cs_cat = compute_chisquare_test(
        df, df_fake_new, cols=discrete_columns, handle_missing=False
    )
    print(cs_cat)


def main_kstest(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns):
    ks_con = compute_kolmogorov_smirnov_test(
        df, df_fake_old, cols=continuous_columns, handle_missing=False
    )
    print(ks_con)

    ks_con = compute_kolmogorov_smirnov_test(
        df, df_fake_new, cols=continuous_columns, handle_missing=False
    )
    print(ks_con)


def main_cramer(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns):
    cs_cat = compute_cramers_v_correlation(df, df_fake_old, cols=discrete_columns)
    print(cs_cat)

    cs_cat = compute_cramers_v_correlation(df, df_fake_new, cols=discrete_columns)
    print(cs_cat)


def main_pearson(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns):
    cs_cat = compute_pearson_correlation(df, df_fake_old, cols=continuous_columns)
    print(cs_cat)

    cs_cat = compute_pearson_correlation(df, df_fake_new, cols=continuous_columns)
    print(cs_cat)


def init():
    D = AdultDataset()
    discrete_columns = D.discrete_columns
    continuous_columns = list(set(D.columns) - set(D.categorical_columns))

    folder = "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_0-lossdwp_0-condvec_1"
    # folder = "adult-tvae-bs_16000-epochs_3000-ed_128-losscorr_0-lossdwp_0-condvec_1"

    is_inverse_transform = True

    label_encoder_path = os.path.join("database/dataset/AdultDataset_label_encoder.pkl")
    df = pd.read_csv(
        os.path.join(f"database/gan/{folder}/preprocessed.csv"),
        sep="\t",
        header=0,
        index_col=0,
    )

    print(df)

    folder_old = (
        "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_0-lossdwp_0-condvec_1"
        # "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_0-lossdwp_0-condvec_0"
    )

    filename_old = config.LIST_BEST[folder]
    df_fake_old = pd.read_csv(
        f"database/gan/{folder_old}/{filename_old}",
        sep="\t",
        header=0,
        index_col=0,
    )
    df_fake_old["capital-gain"] = df_fake_old["capital-gain"].clip(lower=0)
    df_fake_old["capital-loss"] = df_fake_old["capital-loss"].clip(lower=0)

    print(df_fake_old)

    folder_new = (
        "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_1-lossdwp_1-condvec_1"
        # "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_0-lossdwp_1-condvec_1"
        # "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_1-lossdwp_0-condvec_1"
        # "adult-ctgan-bs_16000-epochs_3000-ed_128-losscorr_1-lossdwp_1-condvec_0"
    )

    filename_new = config.LIST_BEST[folder]
    df_fake_new = pd.read_csv(
        f"database/gan/{folder_new}/{filename_new}",
        sep="\t",
        header=0,
        index_col=0,
    )
    df_fake_new["capital-gain"] = df_fake_new["capital-gain"].clip(lower=0)
    df_fake_new["capital-loss"] = df_fake_new["capital-loss"].clip(lower=0)
    df_fake_new

    print(df_fake_new)

    return df, df_fake_old, df_fake_new, discrete_columns, continuous_columns


if __name__ == "__main__":
    df, df_fake_old, df_fake_new, discrete_columns, continuous_columns = init()
    main_kl(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns)
    # main_cs(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns)
    # main_kstest(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns)
    # main_cramer(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns)
    # main_pearson(df, df_fake_old, df_fake_new, discrete_columns, continuous_columns)
