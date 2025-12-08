import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def convert_categories_to_numbers(df, col):
    """
    pseudonymize data
    ["MO", "AB", NAN] -> [0,1, ""]
    """
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes
    df[col] = df[col].replace(-1, np.nan)
    return df


def replace_bin_with_first_number(df, col):
    df[col] = (
        df[col].astype(str).str.strip("()[]").str.split(", ").apply(lambda x: x[0])
    )

    # print(df[col])
    return df


def _legacy_preprocess(df):
    for col in [
        "samp_date_bin",
        "qnr_date_bin",
        "birth_date_bin",
        "dx_date_bin",
        "vitalstatus_date_bin",
        "death_date_bin",
        "age_bin",
        "samp_span_bin",
        "qnr_span_bin",
        "life_span_bin",
        "vital_span_bin",
    ]:
        df = replace_bin_with_first_number(df, col)

    # set categorical type
    for col in [
        "id",
        "sex",
        "llkk_txt",
        "llkk_county_letter",
    ]:
        df = convert_categories_to_numbers(df, col)

    # set numerical type
    for col in [
        "bmi",
        "samp_date_bin",
        "qnr_date_bin",
        "birth_date_bin",
        "dx_date_bin",
        "vitalstatus_date_bin",
        "death_date_bin",
        "age_bin",
        "samp_span_bin",
        "qnr_span_bin",
        "life_span_bin",
        "vital_span_bin",
    ]:
        df[col] = df[col].astype("float")

    df["is_vital"] = np.where(df["vital_span_bin"] != -1, 1, 0)
    df["is_dead"] = np.where(df["death_date_bin"] != -1, 1, 0)

    return df


def clean_record(df):
    for col in [
        "samp_date_bin",
        "qnr_date_bin",
        "birth_date_bin",
        "dx_date_bin",
        "vitalstatus_date_bin",
        "death_date_bin",
        "age_bin",
        "samp_span_bin",
        "qnr_span_bin",
        "life_span_bin",
        "vital_span_bin",
    ]:
        df = replace_bin_with_first_number(df, col)

    for col in [
        "bmi",
        "samp_date_bin",
        "qnr_date_bin",
        "birth_date_bin",
        "dx_date_bin",
        "vitalstatus_date_bin",
        "death_date_bin",
        "age_bin",
        "samp_span_bin",
        "qnr_span_bin",
        "life_span_bin",
        "vital_span_bin",
    ]:
        df[col] = df[col].astype("float")

    if "id" in df:
        col = "id"
        df[col] = df[col].astype("string")

    for col in [
        "sex",
        "llkk_txt",
        "llkk_county_letter",
    ]:
        df[col] = df[col].astype("string")

    df["is_vital"] = np.where(df["vital_span_bin"] != -1, 1, 0)
    df["is_dead"] = np.where(df["death_date_bin"] != -1, 1, 0)

    return df


def clean_patient(df):
    for col in ["have_sample", "sample_diff", "diagnoses"]:
        if col in df.columns:
            df = df.drop([col], axis=1)
    for col in df.columns:
        if "diagnose" in col and "age" not in col:
            df = df.drop([col], axis=1)

    for col in df.columns:
        if col in [
            "bmi",
            "age",
            "sample_first",
            "sample_last",
            "num_diagnoses",
            "municipality",
        ]:
            df[col] = df[col].astype("float")
        elif "_age" in col:  # for example, diagnose_8_age ...
            df[col] = df[col].astype("float")
        else:
            df[col] = df[col].astype("string")

    return df


def preprocess_record(df, label_encoder_path="database/record_label_encoder.pickle"):
    df = df.fillna(-1)
    df = clean_record(df)
    if os.path.isfile(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            multi = pickle.load(f)
        df = multi.transform(df)
    else:
        multi = MultiColumnLabelEncoder(columns=list(df.columns))
        df = multi.fit_transform(df)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(multi, f)
    return df


def preprocess_patient(df, label_encoder_path="database/patient_label_encoder.pickle"):
    df = df.fillna(-1)
    df = clean_patient(df)

    # merge all age-columns and label encode
    merge_columns = []
    for col in df.columns:
        if "age" in col:
            merge_columns.append(col)
        if col in ["sample_first", "sample_last"]:
            merge_columns.append(col)
    # merge all age-columns and label encode

    if os.path.isfile(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            multi = pickle.load(f)
        df = multi.transform(df)
    else:
        multi = MultiColumnLabelEncoder(
            columns=list(df.columns), merge_columns=merge_columns
        )
        df = multi.fit_transform(df)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(multi, f)
    return df


def inverse_transform(df, label_encoder_path="database/record_label_encoder.pickle"):
    if os.path.isfile(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            multi = pickle.load(f)
        inv = multi.inverse_transform(df)
    else:
        raise FileNotFoundError(label_encoder_path)
    return inv


class MultiColumnLabelEncoder:
    def __init__(self, columns=None, merge_columns=None):
        self.columns = columns  # array of column names to encode
        self.merge_columns = merge_columns  # array of column names to merge (age, age_treatment, age_icd, ...)

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        # columns = X.columns  # fixed synthetic data w/o id

        if self.merge_columns is not None:
            data = []
            for col in self.merge_columns:
                data.extend(X[col].tolist())
            label_encoder_merge_columns = LabelEncoder().fit(data)

        for col in columns:
            if self.merge_columns is not None and col in self.merge_columns:
                self.encoders[col] = label_encoder_merge_columns
            else:
                self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        # columns = X.columns  # fixed synthetic data w/o id
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
            output[col] = output[col].astype("float")  # run 2x faster than int64
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        # columns = X.columns  # fixed synthetic data w/o id
        for col in columns:
            X[col] = X[col].astype("int")  # convert categories back to int
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

        # for col in columns:
        #     try:
        #         X[col] = X[col].astype("int")  # convert categories back to int
        #         output[col] = self.encoders[col].inverse_transform(X[col])
        #     except:
        #         try:
        #             X[col] = X[col].astype(str)  # convert categories back to str
        #             output[col] = self.encoders[col].inverse_transform(X[col])
        #         except:
        #             raise ValueError(f"{col} contains previously unseen labels")
        # return output


# I don't know who the original author of this function is,
# but you can use this function to reduce memory
# consumption by 60-70%!
def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and
    modify the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(("Memory usage of dataframe is {:.2f}" "MB").format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print(("Memory usage after optimization is: {:.2f}" "MB").format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def reduce_mem(df):
    for col in [
        "id",
        "subproj",
        "sex",
        "vdc",
        "sm_status",
        "sm_yes_no",
        "unthawed_avail",
        "samp_avail",
        "llkk_txt",
        "llkk_county_letter",
        "icd7_code",
        "icd9_code",
        "icdo2_code",
        "icdo3_code",
        "vitalstatus",
        "predict_cohort",
    ]:
        try:
            df[col] = df[col].astype("category")
        except:
            pass

    return df


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def drop_duplicates(df, df_fake=None):
    if df_fake is None:
        df1 = df.drop_duplicates()
        df1.reset_index(drop=True, inplace=True)
        return df1
    else:
        df1 = df.copy()
        df2 = df_fake.copy()
        df_dup = pd.merge(df1, df2, on=list(df1.columns), how="inner")
        df2 = df2.append(df_dup)
        df2["Duplicated"] = df2.duplicated(
            keep=False
        )  # keep=False marks the duplicated row with a True
        df_safe = df2[~df2["Duplicated"]]  # selects only rows which are not duplicated.
        del df_safe["Duplicated"]  # delete the indicator column
        df_safe.reset_index(drop=True, inplace=True)
        return df_safe


def get_metadata(dataset) -> dict:
    d = {"METADATA_SPEC_VERSION": "SINGLE_TABLE_V1", "columns": {}}
    for col in dataset.data_train:
        if col in dataset.discrete_columns:
            d["columns"][col] = {"sdtype": "categorical"}
        else:
            d["columns"][col] = {"sdtype": "numerical"}

    return d


def sample_df_unique_values(df, n_samples=4000) -> pd.DataFrame:
    """Sample df such that df_sub has all unique values in each column

    Args:
        df (_type_): _description_
        n_samples (int, optional): _description_. Defaults to 4000.

    Returns:
        pd.DataFrame: _description_
    """
    list_drop = []
    for col in df.columns:
        list_drop.append(df[col].drop_duplicates().index.tolist())
    union_list = list(set.intersection(*map(set, list_drop)))

    df_sub = df.loc[union_list]

    while len(df_sub) < n_samples:
        new_row = df.sample(n=1)
        if not new_row.isin(df_sub).all().any():
            df_sub = pd.concat([df_sub, new_row])

    return df_sub


def get_epochs_max_and_max_trials(dataset_name, dict_datasets):
    """
    Returns the epochs_max and max_trials for a given dataset.

    Parameters:
        dataset_name (str): The name of the dataset.
        dict_datasets (dict): The dictionary containing dataset information.

    Returns:
        tuple: A tuple containing (epochs_max, max_trials).
    """
    try:
        for size, details in dict_datasets.items():
            if dataset_name in details["dataset"]:
                return details["epochs_max"], details["max_trials"]

        raise ValueError(
            f"Dataset '{dataset_name}' not found in the provided dictionary."
        )
    except:
        return 2000, 30


# ===========================================================================
# subsampling
# ===========================================================================
def subsample_dataframe_with_column_shuffle_and_target(
    df,
    row_numbers,
    column_percents,
    target_column,
):
    """
    Generate subsets of a pandas DataFrame by subsampling rows and shuffling
    columns before sampling, ensuring the target column is always included.
    The columns are reordered back to their original order.

    Parameters:
        df (pd.DataFrame): The original DataFrame to subsample.
        row_numbers (list): List of percentages (0-100) for row subsampling.
        column_percents (list): List of percentages (0-100) for column subsampling.
        target_column (str): The name of the target column to always include.

    Returns:
        dict: A dictionary where keys are (row_number, column_percent) tuples and
              values are the corresponding subsampled DataFrames.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the DataFrame.")

    subsampled_dfs = {}

    # Convert percentages to proportions
    row_proportions = [p / 100 for p in row_numbers]
    column_proportions = [p / 100 for p in column_percents]

    # Perform subsampling
    for row_prop in row_proportions:
        for col_prop in column_proportions:
            # Subsample rows
            sampled_rows = df.sample(frac=row_prop, random_state=42)

            # Shuffle columns excluding the target column
            non_target_columns = [
                col for col in sampled_rows.columns if col != target_column
            ]
            np.random.seed(42)  # Ensure reproducibility
            np.random.shuffle(non_target_columns)

            # Subsample columns, ensuring the target column is included
            num_columns = int(len(non_target_columns) * col_prop)
            sampled_columns = non_target_columns[:num_columns] + [target_column]

            # Reorder columns to match the original order
            sampled_columns_sorted = sorted(sampled_columns, key=list(df.columns).index)

            # Create the subsampled DataFrame
            subsampled_df = sampled_rows[sampled_columns_sorted]
            subsampled_dfs[(int(row_prop * 100), int(col_prop * 100))] = subsampled_df

    return subsampled_dfs


def subsample_dataframe_by_fixed_rows(
    df,
    row_numbers,
    target_column=None,
):
    """
    Generate subsets of a pandas DataFrame by subsampling a fixed number of rows.
    Optionally ensures the target column is present.

    Parameters:
        df (pd.DataFrame): The original DataFrame to subsample.
        row_numbers (list): List of fixed numbers of rows for subsampling.
        target_column (str, optional): The name of the target column to always include.
                                        If None, no specific column is ensured.

    Returns:
        dict: A dictionary where keys are row counts and values are the corresponding
              subsampled DataFrames.
    """
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the DataFrame.")

    subsampled_dfs = {}

    # Perform subsampling for each fixed row count
    for row_number in row_numbers:
        # If row_number exceeds the total rows in df, take the entire DataFrame
        sampled_row_number = min(row_number, len(df))

        # Subsample rows
        sampled_rows = df.sample(n=sampled_row_number, random_state=42)

        # Ensure target column is included if specified
        if target_column:
            if target_column not in sampled_rows.columns:
                raise ValueError(
                    f"Target column '{target_column}' is missing in subsample."
                )
            sampled_rows = sampled_rows[
                [target_column]
                + [col for col in sampled_rows.columns if col != target_column]
            ]

        # Store the subsampled DataFrame
        subsampled_dfs[sampled_row_number] = sampled_rows

    return subsampled_dfs


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    x = np.random.choice(100_000, 100)
    y = unpackbits(x, 16)
    print(y)
    # _x =
