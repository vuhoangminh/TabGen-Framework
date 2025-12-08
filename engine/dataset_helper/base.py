import os
import numpy as np
import pandas as pd
import json
from PIL import Image
from pandas.core.api import DataFrame as DataFrame
from scipy import ndimage
import re
import pickle
import gzip
import requests
import shutil

import models.tab_ddpm.lib as lib

import abc
from abc import abstractmethod
from enum import Enum

from rich import print
from sklearn.model_selection import train_test_split
from engine.utils.data_utils import MultiColumnLabelEncoder
import engine.utils.path_utils as path_utils
import engine.utils.data_utils as data_utils
from sklearn.preprocessing import LabelEncoder
from engine.config import config


from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


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


# Reuse the BinaryColumnEncoder defined earlier
class BinaryColumnEncoder:
    """
    Learns mapping rules to convert various binary column representations
    into standardized '0'/'1' strings, and can invert back.
    """

    def __init__(self, binary_cols: list):
        self.binary_cols = binary_cols
        self.mapping_ = {}
        self.inverse_mapping_ = {}

    def fit(self, df: pd.DataFrame):
        for col in self.binary_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in DataFrame")
            series = df[col]
            vals = pd.Series(series.dropna().unique()).astype(str)
            val_set = set(vals.tolist())

            if pd.api.types.is_numeric_dtype(series) and val_set.issubset({"0", "1"}):
                m = {"0": "0", "1": "1"}
            elif val_set == {"0", "1"}:
                m = {"0": "0", "1": "1"}
            elif "-1" in val_set and len(val_set) == 2:
                other = (val_set - {"-1"}).pop()
                m = {"-1": "0", other: "1"}
            elif val_set == {"f", "m"}:
                m = {"f": "0", "m": "1"}
            else:
                raise ValueError(f"Column '{col}' has unsupported values {val_set}")

            inv = {v: k for k, v in m.items()}
            self.mapping_[col] = m
            self.inverse_mapping_[col] = inv
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, m in self.mapping_.items():
            out[col] = out[col].astype(str).map(m).astype(object)
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


class EvaluatedDataset(abc.ABC):
    def __init__(self, notebook_path=None, verbose=False):
        self.verbose = verbose
        self.notebook_path = notebook_path

        folder = self._get_dataset_folder()[0]

        self.pkl_path = "database/dataset/{}/{}_index.pkl".format(
            folder, self._get_class_name()
        )
        self.label_encoder_path = "database/dataset/{}/{}_label_encoder.pkl".format(
            folder, self._get_class_name()
        )

        self.pkl_path = self._get_path(self.pkl_path)
        self.label_encoder_path = self._get_path(self.label_encoder_path)

        self.discrete_columns = []

    def _categorize_columns_from_input(self, df):
        cont_cols, date_cols, str_cols, mixed_cols = [], [], [], []

        for col in df.columns:
            s = df[col]
            nonnull = s.dropna()

            # 1) If entirely NaN → mixed
            if nonnull.empty:
                mixed_cols.append(col)
                continue

            # 2) Already datetime dtype?
            if pd.api.types.is_datetime64_any_dtype(s):
                date_cols.append(col)
                continue

            # 3) Boolean columns count as numeric
            if pd.api.types.is_bool_dtype(s):
                cont_cols.append(col)
                continue

            # 4) All non-null are strings?
            if nonnull.map(lambda x: isinstance(x, str)).all():
                # Try parse as date
                try:
                    pd.to_datetime(nonnull, errors="raise", infer_datetime_format=True)
                    date_cols.append(col)
                except Exception:
                    str_cols.append(col)
                continue

            # 5) Pure numeric?
            try:
                pd.to_numeric(nonnull, errors="raise")
                cont_cols.append(col)
                continue
            except Exception:
                pass

            # 6) Anything else → mixed
            mixed_cols.append(col)

        return cont_cols, date_cols, str_cols, mixed_cols

    def categorize_columns(self, df: pd.DataFrame):
        binary_cols, discrete_cols, cont_cols = [], [], []
        for col in df.columns:
            n_unique = df[col].dropna().nunique()
            if n_unique == 2:
                binary_cols.append(col)
            elif self.type_columns[col] == "discrete":
                discrete_cols.append(col)
            else:
                cont_cols.append(col)
        return binary_cols, discrete_cols, cont_cols

    def _get_dataset_folder(self):
        return [
            key
            for key, val in DATASET_CLASSES.items()
            if val == self.__class__.__name__
        ]

    def _get_class_name(self):
        return self.__class__.__name__

    def _get_path(self, path):
        if self.notebook_path is not None:
            return os.path.join(self.notebook_path, path)
        else:
            return path

    def _copy_folder(self, src, dst):
        # Ensure the destination directory doesn't exist, or remove it if it does
        if os.path.exists(dst):
            return

        # Copy the entire directory tree
        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            print(f"{src} not exists. Ignore!")

    def _copy_file(self, src_file, dst_dir):
        os.makedirs(dst_dir, exist_ok=True)  # Ensure the destination directory exists
        shutil.copy(src_file, os.path.join(dst_dir, os.path.basename(src_file)))

    @abstractmethod
    def _read_train(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _read_test(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _get_type_columns(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _setup_task(self, *args, **kwargs) -> list:
        raise NotImplementedError

    def _remove_gz_extension(self, path):
        if path.endswith(".gz"):
            # Remove the .gz extension
            return path[:-3]
        else:
            # Return the original path
            return path

    def _download_url_unzip(self, file_url, filename_path):
        r = requests.get(file_url, allow_redirects=True)
        open(filename_path, "wb").write(r.content)

        # Unzip the file if necessary
        if filename_path.endswith(".gz"):
            with gzip.open(filename_path, "rb") as f, open(
                self._remove_gz_extension(filename_path), "wb"
            ) as outf:
                shutil.copyfileobj(f, outf)

                # Delete the zipped file
                os.remove(filename_path)

    def _download(self, path, dataset_url, filenames):
        # Define the URL base for the dataset files

        # Download all the files
        for filename in filenames:
            print(f">> downloading {filename}")
            # Construct the full URL
            file_url = dataset_url + filename

            # Download the file
            filename_path = path + filename
            self._download_url_unzip(file_url, filename_path)

    def _drop_duplicates(self, verbose=0):
        if verbose:
            print(">> drop duplicates")

        if verbose:
            print(f"before drop train: {len(self.data_train)}")
        self.data_train = self.data_train.drop_duplicates()

        if verbose:
            print(f"after drop train: {len(self.data_train)}")

        if verbose:
            print(f"before drop test: {len(self.data_test)}")
        self.data_test = self.data_test.drop_duplicates()

        if verbose:
            print(f"after drop test: {len(self.data_test)}")

    def _split_df_save_index(self, test_size=0.2, is_test=False):
        """
        Splits a Pandas DataFrame into training and testing sets, saves the indices of both sets to a single file, and returns the training and testing sets.

        Parameters:
        df (pandas.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the DataFrame to use for the testing set.

        Returns:
        (pandas.DataFrame, pandas.DataFrame): The training and testing sets, respectively.
        """

        # Check if the index file exists
        if is_test:
            self.pkl_path = "database/dataset/test_{}_index.pkl".format(
                self._get_class_name()
            )

        try:
            # If the file exists, load the indices
            with open(self.pkl_path, "rb") as f:
                index_df = pickle.load(f)

                # Split the DataFrame using the loaded indices
                self.data_train = self.data.iloc[index_df["train_index"]]
                self.data_test = self.data.iloc[index_df["test_index"]]

                print("Loaded indices from file.")
        except FileNotFoundError:
            # If the file doesn't exist, split the DataFrame and save the indices
            self.data_train, self.data_test = train_test_split(
                self.data, test_size=test_size
            )

            # Save both indices to a file
            indices = {
                "train_index": self.data_train.index,
                "test_index": self.data_test.index,
            }
            with open(self.pkl_path, "wb") as f:
                pickle.dump(indices, f)

                print("Saved indices to file.")

    def _encode_label(self, merge_columns=None) -> pd.DataFrame:
        self.data = pd.concat([self.data_train, self.data_test], axis=0)
        if os.path.isfile(self.label_encoder_path):
            with open(self.label_encoder_path, "rb") as f:
                multi = pickle.load(f)
            self.data = multi.transform(self.data)
        else:
            multi = MultiColumnLabelEncoder(
                columns=self.discrete_columns, merge_columns=merge_columns
            )
            self.data = multi.fit_transform(self.data)
            with open(self.label_encoder_path, "wb") as f:
                pickle.dump(multi, f)

        self.data_train = self.data.iloc[: len(self.data_train)]
        self.data_test = self.data.iloc[len(self.data_train) :]

    def _setup_task(self):
        for key in self.data.columns:
            value = self.type_columns[key]
            if key != self.target:
                self.features.append(key)

        for key in self.data.columns:
            value = self.type_columns[key]
            if value in ["discrete", "binary"]:
                self.discrete_columns.append(key)

        try:
            self.continuous_columns = [
                c for c in list(self.data) if c not in self.discrete_columns
            ]
        except:
            self.continuous_columns = [
                c for c in list(self.data_train) if c not in self.discrete_columns
            ]

        self.path = self._get_base_path()

    # TODO
    def _get_base_path(self):
        return path_utils.get_parent_dir(self.path_train)

    def _extract_cont_cat_y(self, df, cols, is_include_target=False):
        if not is_include_target:
            cols = [col for col in cols if col != self.target]

        # print("cols in _extract_cont_cat_y", cols)

        if len(cols) > 0:
            df_subset = df[cols]
            # print("df_subset in _extract_cont_cat_y", df_subset)
            return df_subset.to_numpy()
        else:
            return None

    def _prep_tabsyn(self):

        def get_column_name_mapping(
            data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
        ):

            if not column_names:
                column_names = np.array(data_df.columns.tolist())

            idx_mapping = {}

            curr_num_idx = 0
            curr_cat_idx = len(num_col_idx)
            curr_target_idx = curr_cat_idx + len(cat_col_idx)

            for idx in range(len(column_names)):

                if idx in num_col_idx:
                    idx_mapping[int(idx)] = curr_num_idx
                    curr_num_idx += 1
                elif idx in cat_col_idx:
                    idx_mapping[int(idx)] = curr_cat_idx
                    curr_cat_idx += 1
                else:
                    idx_mapping[int(idx)] = curr_target_idx
                    curr_target_idx += 1

            inverse_idx_mapping = {}
            for k, v in idx_mapping.items():
                inverse_idx_mapping[int(v)] = k

            idx_name_mapping = {}

            for i in range(len(column_names)):
                idx_name_mapping[int(i)] = column_names[i]

            return idx_mapping, inverse_idx_mapping, idx_name_mapping

        # prepare info.json
        info_path = os.path.join(self.path, f"tabsyn_info.json")
        class_name = self._get_class_name().replace("Dataset", "")

        discrete_columns = [c for c in self.discrete_columns if c != self.target]
        continuous_columns = [
            c
            for c in list(self.data_train)
            if c not in self.discrete_columns and c != self.target
        ]

        info = {
            "name": f"{class_name}",
            "task_type": (
                "regression" if self.output == "regression" else "binclass"
            ),  # binclass or regression,
            "header": "infer",
            "column_names": list(self.data_train.columns),
            "num_col_idx": [
                self.data_train.columns.get_loc(col) for col in continuous_columns
            ],
            "cat_col_idx": [
                self.data_train.columns.get_loc(col) for col in discrete_columns
            ],
            "target_col_idx": [
                self.data_train.columns.get_loc(col) for col in [self.target]
            ],
            "file_type": "csv",
            "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv",
            "test_path": None,
        }

        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]
        column_names = info["column_names"]

        idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
            self.data_train, num_col_idx, cat_col_idx, target_col_idx, column_names
        )

        num_columns = [column_names[i] for i in num_col_idx]
        cat_columns = [column_names[i] for i in cat_col_idx]
        target_columns = [column_names[i] for i in target_col_idx]

        col_info = {}

        swapped_idx_name_mapping = {
            value: key for key, value in idx_name_mapping.items()
        }
        train_df = self.data_train.rename(columns=swapped_idx_name_mapping)
        test_df = self.data_test.rename(columns=swapped_idx_name_mapping)

        for col_idx in num_col_idx:
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())

        for col_idx in cat_col_idx:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

        for col_idx in target_col_idx:
            if info["task_type"] == "regression":
                col_info[col_idx] = {}
                col_info["type"] = "numerical"
                col_info["max"] = float(train_df[col_idx].max())
                col_info["min"] = float(train_df[col_idx].min())
            else:
                col_info[col_idx] = {}
                col_info["type"] = "categorical"
                col_info["categorizes"] = list(set(train_df[col_idx]))

        info["column_info"] = col_info

        train_df = self.data_train.copy()
        test_df = self.data_test.copy()

        for col in num_columns:
            train_df.loc[train_df[col] == "?", col] = np.nan
        for col in cat_columns:
            train_df.loc[train_df[col] == "?", col] = "nan"
        for col in num_columns:
            test_df.loc[test_df[col] == "?", col] = np.nan
        for col in cat_columns:
            test_df.loc[test_df[col] == "?", col] = "nan"

        train_df[num_columns] = train_df[num_columns].astype(np.float32)
        test_df[num_columns] = test_df[num_columns].astype(np.float32)

        info["column_names"] = column_names
        info["train_num"] = train_df.shape[0]
        info["test_num"] = test_df.shape[0]

        info["idx_mapping"] = idx_mapping
        info["inverse_idx_mapping"] = inverse_idx_mapping
        info["idx_name_mapping"] = idx_name_mapping

        metadata = {"columns": {}}
        task_type = info["task_type"]
        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]

        for i in num_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

        for i in cat_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

        if task_type == "regression":
            for i in target_col_idx:
                metadata["columns"][i] = {}
                metadata["columns"][i]["sdtype"] = "numerical"
                metadata["columns"][i]["computer_representation"] = "Float"
        else:
            for i in target_col_idx:
                metadata["columns"][i] = {}
                metadata["columns"][i]["sdtype"] = "categorical"

        info["metadata"] = metadata

        with open(info_path, "w") as fp:
            json.dump(info, fp, indent=4, separators=(",", ": "))

    def _prep_tabddpm(self):
        # Split the DataFrame
        df_train, df_val = train_test_split(
            self.data_train, test_size=0.1, random_state=42
        )
        df_test = self.data_test
        d = {
            "train": df_train,
            "val": df_val,
            "test": df_test,
        }
        for split, df in d.items():
            X_num = self._extract_cont_cat_y(
                df, self.continuous_columns, is_include_target=False
            )
            path = os.path.join(self.path, f"X_num_{split}")
            if X_num is not None and not os.path.exists(path + ".npy"):
                X_num = X_num.astype(np.float32)
                print(f">> save X_num to {path} with shape", X_num.shape)
                np.save(path, X_num)
            if X_num is not None:
                n_num_features = X_num.shape[1]
            else:
                n_num_features = 0

            X_cat = self._extract_cont_cat_y(
                df, self.discrete_columns, is_include_target=False
            )
            path = os.path.join(self.path, f"X_cat_{split}")
            if X_cat is not None and not os.path.exists(path + ".npy"):
                X_cat = X_cat.astype(str)
                print(f">> save X_cat to {path} with shape", X_cat.shape)
                np.save(path, X_cat)
            if X_cat is not None:
                n_cat_features = X_cat.shape[1]
            else:
                n_cat_features = 0

            path = os.path.join(self.path, f"y_{split}")
            y = self._extract_cont_cat_y(df, self.target, is_include_target=True)
            if y is not None and not os.path.exists(path + ".npy"):
                y = (
                    y.astype(np.float32)
                    if self.output == "regression"
                    else y.astype(np.int64)
                )
                print(f">> save y to {path} with shape", y.shape)
                np.save(path, y)

        # prepare info.json
        info_path = os.path.join(self.path, f"info.json")
        class_name = self._get_class_name().replace("Dataset", "")

        d = {
            "name": f"{class_name}",
            "id": f"{class_name}--default",
            "n_num_features": n_num_features,
            "n_cat_features": n_cat_features,
            "train_size": df_train.shape[0],
            "val_size": df_val.shape[0],
            "test_size": df_test.shape[0],
        }
        if not os.path.exists(info_path):
            if self.output == "regression":
                d["task_type"] = "regression"
            else:
                if len(np.unique(y)) > 2:
                    d["task_type"] = "multiclass"
                    d["n_classes"] = len(np.unique(y))
                elif len(np.unique(y)) == 2:
                    d["task_type"] = "binclass"
                else:
                    raise ValueError
            with open(info_path, "w") as fp:
                json.dump(d, fp, indent=4, separators=(",", ": "))

    def _prep_tabddpm_config_toml_mlp(self):
        base_config = lib.load_config(self._get_path("database/dataset/config.toml"))
        base_config["parent_dir"] = ""

        cwd = os.getcwd()
        base_config["real_data_path"] = self.path.replace(cwd + "/", "")

        path_X_num_train = os.path.join(self.path, f"X_num_train.npy")
        path_y_train = os.path.join(self.path, f"y_train.npy")

        y = np.load(path_y_train, allow_pickle=True)
        if os.path.exists(path_X_num_train):
            x = np.load(path_X_num_train, allow_pickle=True)
            base_config["num_numerical_features"] = x.shape[1]
        else:
            base_config["num_numerical_features"] = 0

        if self.output == "regression":
            base_config["model_params"]["is_y_cond"] = False
            base_config["model_params"]["num_classes"] = 0
        else:
            base_config["model_params"]["is_y_cond"] = True
            base_config["model_params"]["num_classes"] = len(np.unique(y))

        base_config["sample"]["num_samples"] = y.shape[0]

        lib.dump_config(base_config, os.path.join(self.path, "config.toml"))

        """
        - prepare X_num_{train,val,test} X_cat_{train,val,test} and y_{train,val,test}
        - config.toml for each dataset
        - prepare info.json -- look at buddy, california, cardio
            {
                "task_type": "multiclass",
                "name": "buddy",
                "id": "buddy--id",
                "n_classes": 3, # only multiclass
                "train_size": 12053,
                "val_size": 3014,
                "test_size": 3767,2647182
                "n_num_features": 4,
                "n_cat_features": 5
            }

            {
                "name": "California Housing",
                "id": "california--default",
                "task_type": "regression",
                "n_num_features": 8,
                "n_cat_features": 0,
                "train_size": 13209,
                "val_size": 3303,
                "test_size": 4128
            }

            {
                "name": "Cardio",
                "id": "cardio--default",
                "task_type": "binclass",
                "n_num_features": 5,
                "n_cat_features": 6,
                "test_size": 11200,
                "train_size": 44800,
                "val_size": 14000
            }
        """

    def _prep_tabddpm_config_toml_resnet(self):
        base_config = lib.load_config(self._get_path("database/dataset/config.toml"))
        base_config["parent_dir"] = ""

        cwd = os.getcwd()
        base_config["real_data_path"] = self.path.replace(cwd + "/", "")

        path_X_num_train = os.path.join(self.path, f"X_num_train.npy")
        path_y_train = os.path.join(self.path, f"y_train.npy")

        y = np.load(path_y_train, allow_pickle=True)
        if os.path.exists(path_X_num_train):
            x = np.load(path_X_num_train, allow_pickle=True)
            base_config["num_numerical_features"] = x.shape[1]
        else:
            base_config["num_numerical_features"] = 0

        if self.output == "regression":
            base_config["model_params"]["is_y_cond"] = False
            base_config["model_params"]["num_classes"] = 0
        else:
            base_config["model_params"]["is_y_cond"] = True
            base_config["model_params"]["num_classes"] = len(np.unique(y))

        base_config["sample"]["num_samples"] = y.shape[0]

        lib.dump_config(base_config, os.path.join(self.path, "config_resnet.toml"))


class TabDDPMDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=True, is_overwrite=False):
        super().__init__(notebook_path)

        self.target = "y"
        self.output = self._get_output()  # Override in derived classes
        self.features = []
        self.discrete_columns = []

        self.path = self._get_path(self._get_base_path())  # Override in derived classes
        self.path_tabddpm = self._get_tabddpm_data_path(self.path)

        if is_overwrite:
            self.data_train, self.data_test, n_con_features, n_dis_features = (
                self._read_npy(self.path_tabddpm)
            )
        else:
            self.data_train, self.data_test, n_con_features, n_dis_features = (
                self._read_npy(self.path)
            )

        self.type_columns = self._get_type_columns(
            self.data_train, n_con_features, n_dis_features
        )

        self.columns = list(self.type_columns.keys())
        self._setup_task()

        if is_overwrite:
            if is_encode:
                self._encode_label()
            X_num_train, X_cat_train, y_train = self._split_dataframe(
                self.data_train, n_con_features, n_dis_features, 1
            )
            X_num_test, X_cat_test, y_test = self._split_dataframe(
                self.data_test, n_con_features, n_dis_features, 1
            )
            self._save_encoded_npy(
                "train", X_num_train, X_cat_train.astype(str).astype(object), y_train
            )
            self._save_encoded_npy(
                "val", X_num_test, X_cat_test.astype(str).astype(object), y_test
            )
            self._save_encoded_npy(
                "test", X_num_test, X_cat_test.astype(str).astype(object), y_test
            )

        self._prep_ctab()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _save_encoded_npy(self, split, X_num, X_cat, y):
        if X_num.shape[1] > 0:
            np.save(os.path.join(self.path, f"X_num_{split}"), X_num)
        if X_cat.shape[1] > 0:
            np.save(os.path.join(self.path, f"X_cat_{split}"), X_cat)
        np.save(os.path.join(self.path, f"y_{split}"), y)

    def _get_base_path(self):
        raise NotImplementedError("Derived classes must implement this method")

    def _get_tabddpm_data_path(self, path):
        return path.replace("database/dataset", "database/dataset/tab_ddpm")

    def _get_output(self):
        raise NotImplementedError("Derived classes must implement this method")

    def _read_train(self) -> pd.DataFrame:
        pass

    def _read_test(self) -> pd.DataFrame:
        pass

    def _read_pure_data(self, path, split="train"):
        y = np.load(os.path.join(path, f"y_{split}.npy"), allow_pickle=True)
        X_num = None
        X_cat = None
        if os.path.exists(os.path.join(path, f"X_num_{split}.npy")):
            X_num = np.load(os.path.join(path, f"X_num_{split}.npy"), allow_pickle=True)
        if os.path.exists(os.path.join(path, f"X_cat_{split}.npy")):
            X_cat = np.load(os.path.join(path, f"X_cat_{split}.npy"), allow_pickle=True)

        return X_num, X_cat, y

    def _concat_to_pd(self, X_num, X_cat, y):
        if X_num is None:
            return pd.concat(
                [
                    pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
                    pd.DataFrame(y, columns=["y"]),
                ],
                axis=1,
            )
        if X_cat is not None:
            return pd.concat(
                [
                    pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
                    pd.DataFrame(
                        X_cat,
                        columns=list(
                            range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1])
                        ),
                    ),
                    pd.DataFrame(y, columns=["y"]),
                ],
                axis=1,
            )
        return pd.concat(
            [
                pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
                pd.DataFrame(y, columns=["y"]),
            ],
            axis=1,
        )

    def _concatenate_or_return(self, X_num_train, X_num_val):
        if X_num_train is None:
            return X_num_val
        elif X_num_val is None:
            return X_num_train
        else:
            return np.concatenate((X_num_train, X_num_val), axis=0)

    def _read_npy(self, data_path):
        X_cat, X_num, y = {}, {}, {}
        X_num_train, X_cat_train, y_train = self._read_pure_data(data_path, "train")
        X_num_val, X_cat_val, y_val = self._read_pure_data(data_path, "val")
        X_num_test, X_cat_test, y_test = self._read_pure_data(data_path, "test")

        X_num_train = self._concatenate_or_return(X_num_train, X_num_val)
        X_cat_train = self._concatenate_or_return(X_cat_train, X_cat_val)
        y_train = self._concatenate_or_return(y_train, y_val)

        X_train = self._concat_to_pd(X_num_train, X_cat_train, y_train)
        X_train.columns = [str(_) for _ in X_train.columns]

        X_test = self._concat_to_pd(X_num_test, X_cat_test, y_test)
        X_test.columns = [str(_) for _ in X_test.columns]

        if X_num_train is not None:
            n_con_features = X_num_train.shape[-1]
        else:
            n_con_features = 0

        if X_cat_train is not None:
            n_dis_features = X_cat_train.shape[-1]
        else:
            n_dis_features = 0

        return X_train, X_test, n_con_features, n_dis_features

    def _get_type_columns(self, df, m, p) -> dict:
        # Initialize an empty dictionary
        col_dict = {}

        # Get the total number of columns
        n = len(df.columns)

        # Assign "continuous" to the first m columns
        for i in range(m):
            col_dict[df.columns[i]] = "continuous"

        # Assign "discrete" to the next p columns
        for i in range(m, m + p):
            col_dict[df.columns[i]] = "discrete"

        # Assign to the target column
        if self.output == "regression":
            col_dict[df.columns[-1]] = "continuous"
        else:
            col_dict[df.columns[-1]] = "discrete"

        return col_dict

    def _split_dataframe(self, df, m, p, q):
        # Ensure m + p + q equals to the number of columns in the DataFrame
        n = df.shape[1]
        if m + p + q != n:
            raise ValueError(
                "m + p + q must equal the number of columns in the dataframe"
            )

        # Split the DataFrame into three parts based on m, p, and q
        df_m = df.iloc[:, :m]  # First m columns
        df_p = df.iloc[:, m : m + p]  # Next p columns
        df_q = df.iloc[:, m + p : m + p + q]  # Last q columns

        # Convert each part to numpy arrays
        array_m = df_m.to_numpy()
        array_p = df_p.to_numpy()
        array_q = df_q.to_numpy()

        return array_m, array_p, np.squeeze(array_q, axis=-1)

    def _prep_ctab(self):
        filename = "database/dataset/ctab_columns.json"
        with open(filename, "r") as f:
            ctab_columns = json.load(f)
        ctabgan_params = ctab_columns[
            path_utils.get_filename_without_extension(self.path)
        ]

        self.general_columns = []
        for key, value in ctabgan_params.items():
            # Set the attribute on the instance
            setattr(self, key, value)
        self.non_categorical_columns = []
        self.log_columns = []
