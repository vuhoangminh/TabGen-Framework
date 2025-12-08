import numpy as np
import pandas as pd
import joblib
from pandas.core.api import DataFrame as DataFrame
from engine.config import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas.api.types as pd_types  # Import for robust type checking


DATASET_CLASSES = config.DATASET_CLASSES


class PCAProcessor:
    def __init__(self, n_components=None, plot_pca=True, var_threshold=0.8):
        """
        Args:
            n_components: Fixed number of PCA components. If None, choose to explain var_threshold.
            plot_pca: Whether to plot cumulative explained variance.
            var_threshold: Fraction of variance to explain if n_components is None.
        """
        self.n_components = n_components
        self.plot_pca = plot_pca
        self.var_threshold = var_threshold
        self.preprocessor = None
        self.pca = None

    def fit_transform(
        self, df: pd.DataFrame, cont_cols, discrete_cols, ordinal_cols, binary_cols
    ):
        # Build preprocessing pipelines
        # Continuous: Quantile transformer to Gaussianize marginals
        cont_pipeline = Pipeline(
            [("qt", QuantileTransformer(output_distribution="normal", random_state=0))]
        )

        # Discrete: one-hot encode
        discrete_pipeline = Pipeline(
            [("onehot", OneHotEncoder(sparse_output=True, handle_unknown="ignore"))]
        )

        # Ordinal and binary: passthrough
        ordinal_pipeline = "passthrough"
        binary_pipeline = "passthrough"

        # Combine
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cont", cont_pipeline, cont_cols),
                ("disc", discrete_pipeline, discrete_cols),
                ("ord", ordinal_pipeline, ordinal_cols),
                ("bin", binary_pipeline, binary_cols),
            ],
            remainder="drop",
        )
        X = self.preprocessor.fit_transform(df)

        # Determine n_components by variance threshold if needed
        if self.n_components is None:
            pca_full = PCA()
            _ = pca_full.fit_transform(X)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = np.searchsorted(cumvar, self.var_threshold) + 1

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)

        # Plot explained variance
        if self.plot_pca:
            cumvar = np.cumsum(self.pca.explained_variance_ratio_)
            plt.figure(figsize=(8, 5))
            plt.plot(cumvar, marker="o")
            plt.axhline(self.var_threshold, color="red", linestyle="--")
            plt.xlabel("Number of components")
            plt.ylabel("Cumulative explained variance")
            plt.grid(True)
            plt.show()

        return X_pca

    def inverse_transform(
        self,
        X_pca: np.ndarray,
        df: pd.DataFrame,
        cont_cols,
        discrete_cols,
        ordinal_cols,
        binary_cols,
    ):
        # Inverse PCA
        X_rec = self.pca.inverse_transform(X_pca)

        # Reconstruct transformed DataFrame
        ohe = self.preprocessor.named_transformers_["disc"].named_steps["onehot"]
        disc_ohe_cols = ohe.get_feature_names_out(discrete_cols)
        cols_transformed = (
            list(cont_cols) + list(disc_ohe_cols) + ordinal_cols + binary_cols
        )
        df_trans = pd.DataFrame(X_rec, columns=cols_transformed)

        # Inverse preprocessing
        result = pd.DataFrame()
        # Continuous: inverse quantile
        qt = self.preprocessor.named_transformers_["cont"].named_steps["qt"]
        cont_quant = df_trans[cont_cols]
        continuous_orig = qt.inverse_transform(cont_quant)
        result[cont_cols] = pd.DataFrame(continuous_orig, columns=cont_cols)

        # Discrete: inverse one-hot
        if discrete_cols:
            disc_block = df_trans[disc_ohe_cols]
            disc_vals = ohe.inverse_transform(disc_block)
            result[discrete_cols] = pd.DataFrame(disc_vals, columns=discrete_cols)

        # Ordinal & Binary: passthrough
        if ordinal_cols:
            result[ordinal_cols] = df_trans[ordinal_cols]
        if binary_cols:
            result[binary_cols] = df_trans[binary_cols]

        # Cast to original dtypes
        for col in discrete_cols + binary_cols + ordinal_cols:
            result[col] = result[col].astype(df[col].dtype)

        return result


def clean_string_columns(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """
    For a given list of columns in a DataFrame, if the column is of a string type,
    converts its values to lowercase and removes leading/trailing whitespace.

    Args:
        df: The input pandas DataFrame.
        columns_to_clean: A list of column names (strings) to process.

    Returns:
        The DataFrame with specified string columns cleaned.
    """
    # Create a copy to avoid modifying the original DataFrame directly,
    # unless in-place modification is explicitly desired.
    # If in-place is desired, remove .copy()
    df_cleaned = df.copy()

    for col in columns_to_clean:
        # Check if the column exists in the DataFrame
        if col in df_cleaned.columns:
            # Check if the column's data type is a string type
            # pd_types.is_string_dtype is a robust way to check for various string dtypes
            if pd_types.is_string_dtype(df_cleaned[col].dtype):
                print(f"Cleaning column: '{col}'")
                # Apply lowercase and strip leading/trailing whitespace
                # .str methods handle NaN values gracefully
                df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
            else:
                print(
                    f"Skipping column '{col}': Not a string type (dtype: {df_cleaned[col].dtype})"
                )
        else:
            print(f"Skipping column '{col}': Not found in DataFrame")

    return df_cleaned


def find_mixed_type_columns(df, columns):
    mixed = []
    for col in columns:
        unique_types = set(df[col].dropna().map(type))
        if len(unique_types) > 1:
            print(
                f"[MIXED TYPES] Column '{col}' has types: {[t.__name__ for t in unique_types]}"
            )
            mixed.append(col)
    return mixed


class BaseEncoder:
    def set_type_columns(self, type_columns):
        self.type_columns = type_columns

    def get_type_columns(self):
        return self.type_columns


class DateEncoder(BaseEncoder):
    """
    Encodes date columns as days since a reference date, and can invert them back to Timestamps.
    """

    def __init__(self, date_columns=None, reference_date="1800-01-01", type_columns={}):
        """
        Args:
            date_columns: List of column names to treat as dates.
                          If None, auto-detects any column with 'date' in its name.
            reference_date: String or Timestamp used as day-zero.
        """
        self.date_columns = date_columns
        self.reference_date = pd.Timestamp(reference_date)
        self.type_columns = type_columns

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = (
            self.date_columns
            if self.date_columns is not None
            else [c for c in out.columns if "date" in c.lower()]
        )

        for col in cols:
            if col not in out.columns:
                raise KeyError(f"Column '{col}' not found")
            # parse to datetime (NaT for bad)
            out[col] = pd.to_datetime(out[col], errors="coerce")
            # days since reference
            out[col] = (out[col] - self.reference_date) // pd.Timedelta("1D")
            # coerce to float
            out[col] = out[col].astype(float)
            self.type_columns[col] = "continuous"
        # store which were encoded
        self.encoded_columns = cols
        return out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        reference_date = self.reference_date

        for col in getattr(self, "encoded_columns", []):
            if col not in out.columns:
                continue
            out[col] = out[col].apply(
                lambda days: (
                    pd.NaT
                    if pd.isna(days) or round(days) == -1
                    else reference_date + pd.Timedelta(days=round(days))
                )
            )
        return out


class MissingValueEncoder(BaseEncoder):
    def __init__(self, noise_std: float = 0.05):
        """
        type_columns: dict of {column_name: "discrete" or "continuous"}
        noise_std: standard deviation of Gaussian noise for continuous imputation
        """
        self.noise_std = noise_std
        self.medians = {}
        self.missing_flags = set()  # Track which _missing columns we added

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.data = df.copy()
        for column in self.data.columns:
            col_type = self.type_columns.get(column)

            if col_type == "discrete":
                self.data[column] = self.data[column].fillna("-1")
                self.data[column] = self.data[column].astype(str)

            elif col_type == "continuous":
                if self.data[column].isna().any():
                    self.data[column] = self.data[column].fillna(-1)

                    # Add missingness flag
                    missing_flag_col = f"{column}_missing"
                    self.data[missing_flag_col] = (self.data[column] == -1).astype(int)
                    self.type_columns[missing_flag_col] = "discrete"
                    self.missing_flags.add(missing_flag_col)

                    # Store median before imputation
                    non_missing = self.data[column] != -1
                    median_val = self.data.loc[non_missing, column].median()
                    self.medians[column] = median_val

                    # Impute -1 with median + noise
                    missing_idx = self.data[column] == -1
                    if self.noise_std > 0 and missing_idx.any():
                        noise = np.random.normal(
                            loc=0, scale=self.noise_std, size=missing_idx.sum()
                        )
                        self.data.loc[missing_idx, column] = median_val + noise
                    else:
                        self.data.loc[missing_idx, column] = median_val
        return self.data

    def inverse_transform(self, df_encoded: pd.DataFrame) -> pd.DataFrame:
        df = df_encoded.copy()

        for column, col_type in self.type_columns.items():
            if col_type == "discrete":
                if column in df.columns:
                    df[column] = df[column].replace("-1", np.nan)

            elif col_type == "continuous":
                missing_flag_col = f"{column}_missing"
                if missing_flag_col in df.columns:
                    df.loc[df[missing_flag_col] == 1, column] = np.nan
                    df.loc[df[missing_flag_col] == "1", column] = np.nan
                    df.drop(columns=[missing_flag_col], inplace=True)

        return df

    def set_all_neg1_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace all values equal to -1 (int, float) or "-1" (string) with np.nan across the entire DataFrame.
        """
        return df.applymap(lambda x: np.nan if str(x) == "-1" else x)


class BinaryColumnEncoder(BaseEncoder):
    """
    Learns mapping rules to convert various binary column representations
    into standardized '0'/'1' strings, and can invert back.
    """

    def __init__(self):
        self.mapping_ = {}
        self.inverse_mapping_ = {}

    def fit(self, df: pd.DataFrame):
        binary_cols = []
        for col in df.columns:
            n_unique = df[col].dropna().nunique()
            if n_unique == 2:
                binary_cols.append(col)
        self.binary_cols = binary_cols

        for col in self.binary_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in DataFrame")
            series = df[col]
            vals = pd.Series(series.dropna().unique())
            str_vals = vals.astype(str)
            val_set = set(str_vals.tolist())

            # Standard 0/1 numeric or strings
            if val_set == {"0", "1"}:
                m = {"0": "0", "1": "1"}
            # True/False boolean or string
            elif val_set == {"True", "False"}:
                m = {"False": "0", "True": "1"}
            elif val_set == {"false", "true"}:
                m = {"false": "0", "true": "1"}
            elif val_set == {"FALSE", "TRUE"}:
                m = {"FALSE": "0", "TRUE": "1"}
            # Yes/No common formats
            elif val_set == {"yes", "no"}:
                m = {"no": "0", "yes": "1"}
            elif val_set == {"Yes", "No"}:
                m = {"No": "0", "Yes": "1"}
            elif val_set == {"Y", "N"}:
                m = {"N": "0", "Y": "1"}
            elif val_set == {"f", "m"}:
                m = {"f": "0", "m": "1"}
            # Numeric values with -1/1, or other two values
            elif "-1" in val_set and len(val_set) == 2:
                other = (val_set - {"-1"}).pop()
                m = {"-1": "0", other: "1"}
            else:
                raise ValueError(f"Column '{col}' has unsupported values {val_set}")

            inv = {v: k for k, v in m.items()}
            self.mapping_[col] = m
            self.inverse_mapping_[col] = inv

            self.type_columns[col] = "discrete"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, m in self.mapping_.items():
            out[col] = out[col].astype(str).map(m).astype(int)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df_enc: pd.DataFrame) -> pd.DataFrame:
        """
        For each binary column:
        1) Snap continuous-ish values back to exactly '0' or '1' by nearest.
        2) Map '0'/'1' back to the original labels.
        """
        out = df_enc.copy()
        for col, inv in self.inverse_mapping_.items():
            # 1) Snap: bring floats back to exactly "0" or "1"
            #    If the encoded column contains floats, threshold at 0.5
            series = out[col].astype(float)
            snapped = series.gt(0.5).astype(int).astype(str)
            out[col] = snapped.map(inv)
        return out


class DiscreteEncoder:
    def __init__(self, discrete_columns, label_encoder_path):
        self.discrete_columns = discrete_columns
        self.label_encoder_path = label_encoder_path
        self.encoders = {}  # To store LabelEncoders for each column

    def fit_transform(self, data, merge_columns=None):
        # Create label encoders for each column
        for col in self.discrete_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
            self.encoders[col] = encoder  # Store the encoder for each column
        return data

    def inverse_transform(self, data):
        """Inverse transforms the encoded data to original values."""
        for col in self.discrete_columns:
            if col in self.encoders:
                data[col] = self.encoders[col].inverse_transform(data[col])
        return data


class DataFramePreprocessor(BaseEstimator, TransformerMixin, BaseEncoder):
    """
    A bidirectional preprocessor for tabular data.
    Supports:
      - Continuous: Quantile transformation (Gaussian)
      - Discrete: One-hot encoding (optional passthrough)
      - Ordinal & Binary: Passthrough
    Allows full inverse transformation to original structure and dtypes.
    """

    def __init__(self, use_ohe_for_discrete=True):
        self.preprocessor = None
        self.use_ohe_for_discrete = use_ohe_for_discrete

    def set_columns(self, cont_cols, discrete_cols, ordinal_cols, binary_cols):
        self.cont_cols = cont_cols
        self.discrete_cols = discrete_cols
        self.ordinal_cols = ordinal_cols
        self.binary_cols = binary_cols
        self.original_dtypes = {}

    def update_type_columns(self, discrete_cols, disc_ohe_cols):
        self.type_columns = getattr(self, "type_columns", {})
        for key in discrete_cols:
            if key in self.type_columns:
                del self.type_columns[key]
        for key in disc_ohe_cols:
            self.type_columns[key] = "discrete"
        return self.type_columns

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Store original dtypes for inverse
        for col in self.discrete_cols + self.ordinal_cols + self.binary_cols:
            self.original_dtypes[col] = df[col].dtype

        # Pipelines
        cont_pipeline = Pipeline(
            [("qt", QuantileTransformer(output_distribution="normal", random_state=0))]
        )

        if self.use_ohe_for_discrete:
            discrete_pipeline = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    )
                ]
            )
        else:
            discrete_pipeline = "passthrough"

        ordinal_pipeline = "passthrough"
        binary_pipeline = "passthrough"

        # ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cont", cont_pipeline, self.cont_cols),
                ("disc", discrete_pipeline, self.discrete_cols),
                ("ord", ordinal_pipeline, self.ordinal_cols),
                ("bin", binary_pipeline, self.binary_cols),
            ],
            remainder="drop",
        )

        X_array = self.preprocessor.fit_transform(df)

        # Build column names
        disc_transformer = self.preprocessor.named_transformers_["disc"]
        if isinstance(disc_transformer, Pipeline):
            ohe = disc_transformer.named_steps["onehot"]
            self.disc_ohe_cols = list(ohe.get_feature_names_out(self.discrete_cols))
        elif disc_transformer == "passthrough" or isinstance(
            disc_transformer, FunctionTransformer
        ):
            self.disc_ohe_cols = list(self.discrete_cols)
        else:
            raise ValueError("Unsupported transformer type for discrete columns.")

        cols_transformed = (
            list(self.cont_cols)
            + self.disc_ohe_cols
            + self.ordinal_cols
            + self.binary_cols
        )

        self.update_type_columns(self.discrete_cols, self.disc_ohe_cols)

        X_df = pd.DataFrame(X_array, columns=cols_transformed, index=df.index)
        return X_df

    def inverse_transform(self, X_rec: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Must call fit_transform before inverse_transform")

        df_trans = pd.DataFrame(X_rec, columns=X_rec.columns, index=X_rec.index)
        result = pd.DataFrame(index=df_trans.index)

        # Inverse continuous
        qt = self.preprocessor.named_transformers_["cont"].named_steps["qt"]
        cont_block = df_trans[self.cont_cols]
        cont_inverse = qt.inverse_transform(cont_block)
        result[self.cont_cols] = pd.DataFrame(
            cont_inverse, columns=self.cont_cols, index=df_trans.index
        )

        # Inverse discrete
        disc_transformer = self.preprocessor.named_transformers_["disc"]
        disc_block = df_trans[self.disc_ohe_cols]

        if isinstance(disc_transformer, Pipeline):
            ohe = disc_transformer.named_steps["onehot"]
            disc_inverse = ohe.inverse_transform(disc_block)
            result[self.discrete_cols] = pd.DataFrame(
                disc_inverse, columns=self.discrete_cols, index=df_trans.index
            )
        elif disc_transformer == "passthrough" or isinstance(
            disc_transformer, FunctionTransformer
        ):
            result[self.discrete_cols] = disc_block
        else:
            raise ValueError("Unsupported transformer type for discrete columns.")

        # Ordinal & Binary passthrough
        if self.ordinal_cols:
            result[self.ordinal_cols] = df_trans[self.ordinal_cols]
        if self.binary_cols:
            result[self.binary_cols] = df_trans[self.binary_cols]

        # Restore original dtypes
        for col in self.discrete_cols + self.ordinal_cols + self.binary_cols:
            result[col] = result[col].astype(self.original_dtypes[col])

        return result


class FlexiblePipeline:
    def __init__(self, steps, type_columns=None):
        self.steps = steps
        self.type_columns = type_columns or {}
        self.binary_cols = []
        self.discrete_cols = []
        self.cont_cols = []

    def update_columns(self, df):
        binary_cols, discrete_cols, cont_cols = [], [], []
        for col in df.columns:
            n_unique = df[col].dropna().nunique()
            if n_unique == 2:
                binary_cols.append(col)
            elif self.type_columns[col] == "discrete":
                discrete_cols.append(col)
            else:
                cont_cols.append(col)

        self.binary_cols = binary_cols
        self.discrete_cols = discrete_cols
        self.cont_cols = cont_cols

    def fit_transform(self, X):
        for name, step in self.steps:
            type_columns = self.type_columns.copy()
            if hasattr(step, "set_type_columns"):
                step.set_type_columns(type_columns)
            if name != "preprocess":
                X = step.fit_transform(X)
            else:
                self.update_columns(X)
                step.set_columns(
                    self.cont_cols, self.discrete_cols, [], self.binary_cols
                )
                X = step.fit_transform(X)
            if hasattr(step, "get_type_columns"):
                type_columns = step.get_type_columns()
            self.type_columns = type_columns
            if hasattr(step, "binary_cols"):
                self.binary_cols = step.binary_cols
            print()
            print()
            print()
            print(f"-- after {name}: --")
            mixed = find_mixed_type_columns(X, X.columns)
            print(X)
        return X

    def inverse_transform(self, X):
        for name, step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                X = step.inverse_transform(X)
                print(f"{name}")
                print(X)
        return X

    @staticmethod
    def match_df_columns_and_types(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        exclude_list: list = [],
    ) -> pd.DataFrame:
        """
        Rearranges columns in df1 to match the order of df2 and sets the data type
        of each column in df1 to match the corresponding column in df2.
        Adds missing columns from df2 to df1 (filled with NaN) and drops columns
        from df1 not present in df2.

        Args:
            df1 (pd.DataFrame): The DataFrame to modify.
            df2 (pd.DataFrame): The reference DataFrame for column order and types.

        Returns:
            pd.DataFrame: A new DataFrame based on df1 with columns and types
                        matching df2.
        """
        # Get the desired column order from df2
        desired_column_order = df2.columns.tolist()

        # Identify columns in df1 that are not in df2
        cols_to_drop_from_df1 = [
            col for col in df1.columns if col not in desired_column_order
        ]

        # Identify columns in df2 that are not in df1
        cols_to_add_to_df1 = [
            col for col in desired_column_order if col not in df1.columns
        ]

        # Create a copy of df1 to avoid modifying the original DataFrame in place
        df1_modified = df1.copy()

        # Drop columns from df1 that are not in df2
        if cols_to_drop_from_df1:
            print(f"Dropping columns from df1 not in df2: {cols_to_drop_from_df1}")
            df1_modified = df1_modified.drop(columns=cols_to_drop_from_df1)

        # Add columns to df1 that are in df2 but not in df1
        if cols_to_add_to_df1:
            print(f"Adding columns to df1 from df2: {cols_to_add_to_df1}")
            for col in cols_to_add_to_df1:
                # Add the column, initially filled with NaN
                df1_modified[col] = np.nan
                # Set the dtype of the newly added column to match df2 immediately
                # This is important if the target dtype is non-numeric (like object or category)
                if (
                    col in df2.columns
                ):  # Should always be true based on cols_to_add_to_df1 logic
                    target_dtype = df2[col].dtype
                    try:
                        df1_modified[col] = df1_modified[col].astype(target_dtype)
                    except TypeError as e:
                        print(
                            f"Warning: Could not set initial dtype for new column '{col}' to {target_dtype}. Reason: {e}"
                        )
                        print("Attempting type conversion after reindexing.")

        # Reindex df1_modified to match the column order of df2
        # This will also handle adding columns if they weren't added explicitly above
        # (though explicit adding helps set initial dtypes for non-numeric)
        df1_modified = df1_modified.reindex(columns=desired_column_order)

        # Set the data type of each column in df1_modified to match df2
        print("Matching data types of columns...")
        for col in desired_column_order:
            if col in exclude_list:
                continue

            if col in df1_modified.columns and col in df2.columns and "date" not in col:
                target_dtype = df2[col].dtype
                current_dtype = df1_modified[col].dtype

                if current_dtype != target_dtype:
                    # print(
                    #     f"  Converting column '{col}' from {current_dtype} to {target_dtype}"
                    # )
                    try:
                        # Use errors='coerce' to turn values that cannot be converted into NaN
                        # This is important for robustness, e.g., converting strings to numbers
                        df1_modified[col] = df1_modified[col].astype(
                            target_dtype, errors="ignore"
                        )
                    except Exception as e:
                        # Catching a general exception here as astype can raise various errors
                        print(
                            f"  Warning: Could not convert column '{col}' to {target_dtype}. Reason: {e}"
                        )
                        print(
                            "  Values that failed conversion might be replaced by NaN."
                        )
            elif col not in df1_modified.columns:
                pass
                # print(
                #     f"  Column '{col}' from df2 was added but not found in df1_modified after reindexing. This is unexpected."
                # )

        return df1_modified

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def get_params(self, deep=True):
        return {name: step for name, step in self.steps}

    def set_params(self, **kwargs):
        for name, step in kwargs.items():
            for i, (n, _) in enumerate(self.steps):
                if n == name:
                    self.steps[i] = (name, step)
        return self


def test_preprocessor():
    df = pd.DataFrame(
        {
            "age": [25, 40, 60],
            "gender": ["M", "F", "F"],
            "education": [1, 2, 3],
            "smoker": [0, 1, 0],
        }
    )

    pre = DataFramePreprocessor(
        cont_cols=["age"],
        discrete_cols=["gender"],
        ordinal_cols=["education"],
        binary_cols=["smoker"],
    )

    X = pre.fit_transform(df)
    df_inv = pre.inverse_transform(X)

    print(df)
    print(X)
    print(df_inv)


def test_flexible():
    # Create test data
    df = pd.DataFrame(
        {
            "visit_date": ["2000-01-01", None, "2001-01-01"],
            "age": [25, np.nan, 35],
            "sex": ["M", "F", "M"],
            "bmi": [22.5, 25.0, np.nan],
            "is_smoker": [0, 1, 0],
            "occ": ["a", "b", "c"],
            "t": ["yes", np.nan, np.nan],
        }
    )

    df = clean_string_columns(df, columns_to_clean=df.columns)

    # Define which columns are continuous, etc.
    type_columns = {
        "visit_date": "discrete",
        "age": "continuous",
        "bmi": "continuous",
        "sex": "discrete",
        "is_smoker": "discrete",
        "occ": "discrete",
        "t": "discrete",
    }

    pipeline = FlexiblePipeline(
        [
            ("date", DateEncoder()),
            ("missing", MissingValueEncoder()),
            ("binary", BinaryColumnEncoder()),
            ("preprocess", DataFramePreprocessor(use_ohe_for_discrete=False)),
        ],
        type_columns=type_columns,
    )

    # Apply
    print()
    print()
    print(">> fit_transform..")
    print()
    print()
    df_trans = pipeline.fit_transform(df)

    print()
    print()
    print(">> inverse_transform..")
    print()
    print()
    df_recovered = pipeline.inverse_transform(df_trans)
    df_recovered = pipeline.match_df_columns_and_types(df_recovered, df)

    print()
    print()
    print()
    print()
    print("Original:")
    print(df)
    print("\nTransformed:")
    print(df_trans.head())
    print("\nRecovered:")
    print(df_recovered.head())


if __name__ == "__main__":
    test_flexible()
