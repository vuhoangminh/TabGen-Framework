from engine.dataset_helper.base import *


class AdultDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(notebook_path)

        self.target = "income"
        self.output = "classification"
        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())
        self.features = []
        self.discrete_columns = []

        self.path_train = "database/dataset/adult/adult.data"
        self.path_test = "database/dataset/adult/adult.test"

        self.path_train = self._get_path(self.path_train)
        self.path_test = self._get_path(self.path_test)

        self.data_train = self._read_train()
        self.data_test = self._read_test()

        self._setup_task()
        self._drop_duplicates()
        self._preprocess()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        return pd.read_csv(self.path_train, names=self.columns, header=None)

    def _read_test(self) -> pd.DataFrame:
        return pd.read_csv(self.path_test, names=self.columns, header=None, skiprows=1)

    def _get_type_columns(self) -> dict:
        return {
            "age": "continuous",
            "workclass": "discrete",
            "fnlwgt": "continuous",
            "education": "discrete",
            "education-num": "continuous",
            "marital-status": "discrete",
            "occupation": "discrete",
            "relationship": "discrete",
            "race": "discrete",
            "sex": "binary",
            "capital-gain": "continuous",
            "capital-loss": "continuous",
            "hours-per-week": "continuous",
            "native-country": "discrete",
            "income": "binary",
        }

    def _preprocess(self):
        print(">> preprocessing")

        self.data_train = self.data_train.fillna(-1)
        self.data_train = self.data_train.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        self.data_train.loc[
            self.data_train[self.target].str.contains("<=50K"), self.target
        ] = "<=50K"
        self.data_train.loc[
            self.data_train[self.target].str.contains(">50K"), self.target
        ] = ">50K"

        self.data_test = self.data_test.fillna(-1)
        self.data_test = self.data_test.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        self.data_test.loc[
            self.data_test[self.target].str.contains("<=50K"), self.target
        ] = "<=50K"
        self.data_test.loc[
            self.data_test[self.target].str.contains(">50K"), self.target
        ] = ">50K"

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {"capital-loss": [0.0], "capital-gain": [0.0]}
        self.general_columns = ["age"]
        self.non_categorical_columns = []
        self.integer_columns = [
            "age",
            "fnlwgt",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "education-num",
        ]
        self.problem_type = {"Classification": "income"}


class CencusDataset(AdultDataset):
    def __init__(self):
        super().__init__()

        self.path_train = "database/dataset/census_income/adult.data"
        self.path_test = "database/dataset/census_income/adult.test"
        self.data_train = self._read_train()
        self.data_test = self._read_test()

        self._setup_task()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()


class CovertypeDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(notebook_path)

        self.target = "Cover_Type"
        self.output = "classification"
        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())
        self.features = []
        self.discrete_columns = []

        self.path_train = "database/dataset/covertype/covtype.data"
        self.path_train = self._get_path(self.path_train)
        self.data = self._read_train()

        self._split_df_save_index()
        self._setup_task()

        self._drop_duplicates()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        return pd.read_csv(self.path_train, names=self.columns, header=None)

    def _read_test(self) -> pd.DataFrame:
        pass

    def _get_type_columns(self) -> dict:
        cont_names = [
            "Elevation",
            "Aspect",
            "Slope",
            "R_Hydrology",
            "Z_Hydrology",
            "R_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "R_Fire_Points",
        ]  # Continuous variables

        cont_dict = {name: "continuous" for name in cont_names}
        area_names = ["WArea_" + str(i + 1) for i in range(4)]
        area_dict = {name: "binary" for name in area_names}
        soil_names = ["Soil_" + str(i + 1) for i in range(40)]
        soil_dict = {name: "binary" for name in soil_names}

        target = "Cover_Type"
        target_dict = {target: "discrete"}

        dtypes_dict = {**cont_dict, **area_dict, **soil_dict, **target_dict}
        return dtypes_dict

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {"Hillshade_3pm": [0.0]}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = [
            "Elevation",
            "Aspect",
            "Slope",
            "R_Hydrology",
            "Z_Hydrology",
            "R_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "R_Fire_Points",
        ]
        self.problem_type = {"Classification": "Cover_Type"}


class CreditDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=False):
        super().__init__(notebook_path)

        self.target = "Class"
        self.output = "classification"
        self.features = []
        self.discrete_columns = []

        self.path_train = "database/dataset/credit/creditcard.csv"
        self.path_train = self._get_path(self.path_train)
        self.data = self._read_train()

        self.data_train = self.data.copy()
        self.type_columns = self._get_type_columns()

        self._split_df_save_index()
        self._setup_task()
        self.columns = list(self.type_columns.keys())

        self._drop_duplicates()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_train)
        df = df.drop(["Time"], axis=1)
        return df

    def _read_test(self) -> pd.DataFrame:
        pass

    def _get_type_columns(self) -> dict:
        columns_dict = {
            name: "continuous"
            for name in self.data_train.columns
            if name != self.target
        }
        columns_dict[self.target] = "binary"
        return columns_dict

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {"Amount": [0.0]}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Classification": "Class"}


class IntrusionDataset(EvaluatedDataset):
    def __init__(
        self, is_download=False, is_test=True, notebook_path=None, is_encode=True
    ):
        super().__init__(notebook_path)

        self.is_download = is_download
        self.is_test = is_test

        if self.is_download:
            self._download(
                self,
                path="database/dataset/intrusion/",
                dataset_url="https://kdd.ics.uci.edu/databases/kddcup99/",
                filenames=[
                    "kddcup.names",
                    "kddcup.data.gz",
                    "kddcup.data_10_percent.gz",
                    "kddcup.newtestdata_10_percent_unlabeled.gz",
                    "kddcup.testdata.unlabeled.gz",
                    "kddcup.testdata.unlabeled_10_percent.gz",
                    "corrected.gz",
                    "training_attack_types",
                    "typo-correction.txt",
                ],
            )

        self.target = "outcome"
        self.output = "classification"
        self.features = []
        self.discrete_columns = []

        if self.is_test:
            self.path_train = "database/dataset/intrusion/kddcup.data_10_percent"
        else:
            self.path_train = "database/dataset/intrusion/kddcup.data"
        self.path_test = "database/dataset/intrusion/corrected"

        self.path_train = self._get_path(self.path_train)
        self.path_test = self._get_path(self.path_test)

        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())

        self.data_train = self._read_train()
        self.data_test = self._read_test()

        self._setup_task()
        # self._drop_duplicates()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path_train, names=self.columns, header=None, index_col=False
        )

    def _read_test(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path_test, names=self.columns, header=None, index_col=False
        )

    def _get_type_columns(self) -> dict:
        dataset_column_path = "database/dataset/intrusion/kddcup.names"
        dataset_column_path = self._get_path(dataset_column_path)
        columns_dict = {}

        with open(dataset_column_path, "r") as file:
            column_labels: str = file.read()

        column_regex: re.Pattern = re.compile(
            r"^(?P<column_name>\w+): (?P<data_type>\w+)\.$"
        )
        for column_type in column_labels.splitlines()[1:]:
            match = column_regex.match(column_type)
            key = match.group("column_name")
            value = (
                match.group("data_type")
                if match.group("data_type") == "continuous"
                else "discrete"
            )
            columns_dict[key] = value

        columns_dict[self.target] = "discrete"

        return columns_dict

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns + [
            "wrong_fragment",
            "urgent",
            "num_failed_logins",
            "root_shell",
            "su_attempted",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
        ]
        self.log_columns = []
        self.mixed_columns = {
            "duration": [0.0],
            "src_bytes": [0.0],
            "dst_bytes": [0.0],
            # "wrong_fragment": [0.0],
            # "urgent": [0.0],
            "hot": [0.0],
            # "num_failed_logins": [0.0],
            "num_compromised": [0.0],
            # "root_shell": [0.0],
            # "su_attempted": [0.0],
            "num_root": [0.0],
            "num_file_creations": [0.0],
            # "num_shells": [0.0],
            # "num_access_files": [0.0],
            "count": [511.0],
            "srv_count": [511.0],
            "serror_rate": [0.0],
            "srv_serror_rate": [0.0],
            "rerror_rate": [0.0],
            "srv_rerror_rate": [0.0],
            "same_srv_rate": [0.0],
            "diff_srv_rate": [0.0],
            "srv_diff_host_rate": [0.0],
            "srv_diff_host_rate": [0.0],
            "dst_host_count": [255.0],
            "dst_host_srv_count": [255.0],
            "dst_host_same_srv_rate": [1.0],
            "dst_host_diff_srv_rate": [0.0],
            "dst_host_same_src_port_rate": [1.0],
            "dst_host_srv_diff_host_rate": [0.0],
            "dst_host_serror_rate": [0.0],
            "dst_host_srv_serror_rate": [0.0],
            "dst_host_rerror_rate": [0.0],
            "dst_host_srv_rerror_rate": [0.0],
        }
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        for key, value in self.mixed_columns.items():
            if "rate" not in key:
                self.integer_columns.append(key)
        self.problem_type = {"Classification": "outcome"}


class MNISTDataset(EvaluatedDataset):
    def __init__(
        self, notebook_path=None, is_resize=False, is_binarize=True, is_encode=True
    ):
        super().__init__(notebook_path)

        self.target = "number"
        self.output = "classification"
        self.features = []
        self.discrete_columns = []

        if is_resize:  # mnist12
            dataset = "mnist12"
        else:
            dataset = "mnist28"

        self.path_train = f"database/dataset/{dataset}/train-images.idx3-ubyte"
        self.path_train_label = f"database/dataset/{dataset}/train-labels.idx1-ubyte"
        self.path_test = f"database/dataset/{dataset}/t10k-images.idx3-ubyte"
        self.path_test_label = f"database/dataset/{dataset}/t10k-labels.idx1-ubyte"

        self.path_train = self._get_path(self.path_train)
        self.path_train_label = self._get_path(self.path_train_label)
        self.path_test = self._get_path(self.path_test)
        self.path_test_label = self._get_path(self.path_test_label)

        self.train_images, self.train_labels = self._read_train()
        self.test_images, self.test_labels = self._read_test()

        if is_resize:
            print(">> resizing")
            self.train_images = self._resize(self.train_images)
            self.test_images = self._resize(self.test_images)

        if is_binarize:
            print(">> binarizing")
            self.train_images = np.round(self.train_images / 255)
            self.test_images = np.round(self.test_images / 255)

        # self._visualize(self.train_images, self.train_labels)

        self._prepare_data()

        self.type_columns = self._get_type_columns()
        self._setup_task()
        self.columns = list(self.type_columns.keys())

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _prepare_data(self):
        cols = [
            f"feature_{i:03d}"
            for i in range(self.train_images.shape[1] * self.train_images.shape[2])
        ]

        self.data_train = self.train_images.reshape(
            self.train_images.shape[0],
            self.train_images.shape[1] * self.train_images.shape[2],
        )
        self.train_labels = self.train_labels.reshape(self.train_labels.shape[0], 1)
        self.data_train = np.concatenate((self.data_train, self.train_labels), axis=1)
        self.data_train = pd.DataFrame(self.data_train, columns=cols + [self.target])

        self.data_test = self.test_images.reshape(
            self.test_images.shape[0],
            self.test_images.shape[1] * self.test_images.shape[2],
        )
        self.test_labels = self.test_labels.reshape(self.test_labels.shape[0], 1)
        self.data_test = np.concatenate((self.data_test, self.test_labels), axis=1)
        self.data_test = pd.DataFrame(self.data_test, columns=cols + [self.target])

    def _read_train(self) -> pd.DataFrame:
        # Load the training data
        with open(self.path_train, "rb") as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            num_images = int.from_bytes(f.read(4), byteorder="big")
            num_rows = int.from_bytes(f.read(4), byteorder="big")
            num_cols = int.from_bytes(f.read(4), byteorder="big")
            train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_images, num_rows, num_cols
            )

        with open(self.path_train_label, "rb") as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            num_labels = int.from_bytes(f.read(4), byteorder="big")
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)

        return train_images, train_labels

    def _read_test(self) -> pd.DataFrame:
        # Load the testing data
        with open(self.path_test, "rb") as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            num_images = int.from_bytes(f.read(4), byteorder="big")
            num_rows = int.from_bytes(f.read(4), byteorder="big")
            num_cols = int.from_bytes(f.read(4), byteorder="big")
            test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_images, num_rows, num_cols
            )

        with open(self.path_test_label, "rb") as f:
            magic_number = int.from_bytes(f.read(4), byteorder="big")
            num_labels = int.from_bytes(f.read(4), byteorder="big")
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)

        return test_images, test_labels

    def _resize(self, images):
        resized_images = []
        for image in images:
            image = np.array(image, dtype="float").reshape((28, 28))
            resized_image = ndimage.zoom(image, (12 / 28, 12 / 28), mode="nearest")
            resized_images.append(resized_image)
        resized_images = np.array(resized_images)
        return resized_images

    def _visualize(self, images, labels, n_images=32):
        import matplotlib.pyplot as plt

        random_images = np.random.choice(images.shape[0], n_images, replace=False)
        fig, axes = plt.subplots(nrows=int(n_images / 8), ncols=8, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(
                images[random_images[i]].reshape(images.shape[1], images.shape[2]),
                cmap="gray",
            )
            ax.set_xlabel(labels[random_images[i]])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def _get_type_columns(self) -> dict:
        columns_dict = {
            name: "binary" for name in self.data_train.columns if name != self.target
        }
        columns_dict[self.target] = "discrete"
        return columns_dict

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Classification": "number"}


class MNIST28x28Dataset(MNISTDataset):
    def __init__(self, notebook_path=None):
        super().__init__(notebook_path, is_resize=False)


class MNIST12x12Dataset(MNISTDataset):
    def __init__(self, notebook_path=None):
        super().__init__(notebook_path, is_resize=True)


class DiabetesDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_balanced=False, is_encode=True):
        super().__init__(notebook_path)

        self.target = "Diabetes_binary"
        self.output = "classification"
        self.features = []
        self.discrete_columns = []

        if is_balanced:
            self.path_train = "database/dataset/diabetesbalanced/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
        else:
            self.path_train = "database/dataset/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv"

        self.path_train = self._get_path(self.path_train)
        self.data = self._read_train()
        self.data_train = self.data.copy()
        self.type_columns = self._get_type_columns()

        self._split_df_save_index()
        self._setup_task()
        self.columns = list(self.type_columns.keys())

        self._drop_duplicates()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_train)
        return df

    def _read_test(self) -> pd.DataFrame:
        pass

    def _is_binary(self, df, col):
        # check if col1 has only 2 unique values which are 0 and 1
        unique_values = df[col].unique()
        if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
            return True
        else:
            return False

    def _get_type_columns(self) -> dict:
        columns_dict = {}
        for col in self.data_train:
            if self._is_binary(self.data_train, col):
                columns_dict[col] = "binary"
            else:
                columns_dict[col] = "discrete"

        columns_dict[self.target] = "binary"
        return columns_dict

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Classification": "Diabetes_binary"}


class DiabetesBalancedDataset(DiabetesDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(notebook_path, is_encode=True, is_balanced=True)


class NewsDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(notebook_path)

        self.target = "shares"
        self.output = "regression"
        self.features = []
        self.discrete_columns = []

        self.path_train = "database/dataset/news/OnlineNewsPopularity.csv"
        self.path_train = self._get_path(self.path_train)

        self.data = self._read_train()
        self.data_train = self.data.copy()

        self._split_df_save_index()
        self._preprocess()

        self.type_columns = self._get_type_columns()
        self._setup_task()
        self.columns = list(self.type_columns.keys())

        self._drop_duplicates()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_train)
        df.columns = [col.replace(" ", "") for col in df.columns]
        df = df.drop(labels=["url"], axis=1)
        return df

    def _read_test(self) -> pd.DataFrame:
        pass

    def _is_binary(self, df, col):
        # check if col1 has only 2 unique values which are 0 and 1
        unique_values = df[col].unique()
        if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
            return True
        else:
            return False

    def _get_type_columns(self) -> dict:
        columns_dict = {}
        for col in self.data_train:
            if self._is_binary(self.data_train, col):
                columns_dict[col] = "binary"
            else:
                columns_dict[col] = "continuous"

        columns_dict[self.target] = "continuous"
        return columns_dict

    def _preprocess(self):
        print(">> preprocessing")

        self.data_train.columns = self.data_train.columns.str.replace(" ", "")
        self.data_test.columns = self.data_test.columns.str.replace(" ", "")

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {
            "n_tokens_content": [0.0],
            "n_unique_tokens": [0.0],
            "n_non_stop_words": [0.0],
            "n_non_stop_unique_tokens": [0.0],
            "num_videos": [0.0],
            "average_token_length": [0.0],
            "kw_avg_min": [-1.0],
            "kw_min_max": [0.0],
            "kw_min_avg": [0.0],
            "kw_avg_avg": [0.0],
            "self_reference_min_shares": [0.0],
            "self_reference_max_shares": [0.0],
            "self_reference_avg_sharess": [0.0],
            "global_subjectivity": [0.0],
            "global_sentiment_polarity": [0.0],
            "global_rate_positive_words": [0.0],
            "global_rate_negative_words": [0.0],
            "avg_positive_polarity": [0.0],
            "avg_negative_polarity": [0.0],
            "title_subjectivity": [0.0],
            "title_sentiment_polarity": [0.0],
            "abs_title_sentiment_polarity": [0.0],
        }
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = [
            "timedelta",
            "n_tokens_title",
            "n_tokens_content",
            "num_hrefs",
            "num_self_hrefs",
            "num_imgs",
            "num_videos",
            "num_keywords",
            "kw_min_min",
            "kw_min_max",
            "kw_max_max",
            "shares",
        ]
        self.problem_type = {"Regression": "shares"}


class HouseDataset(EvaluatedDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(notebook_path)

        self.target = "price"
        self.output = "regression"
        self.features = []
        self.discrete_columns = []

        self.path_train = "database/dataset/house/kc_house_data.csv"

        self.path_train = self._get_path(self.path_train)
        self.data = self._read_train()

        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())

        self._split_df_save_index()
        self._setup_task()

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _read_train(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_train, index_col=False)
        df = df.drop(["id", "date"], axis=1)
        return df

    def _read_test(self) -> pd.DataFrame:
        pass

    def _get_type_columns(self) -> dict:
        return {
            "price": "continuous",
            "bedrooms": "discrete",
            "bathrooms": "discrete",
            "sqft_living": "continuous",
            "sqft_lot": "continuous",
            "floors": "discrete",
            "waterfront": "discrete",
            "view": "discrete",
            "condition": "discrete",
            "grade": "discrete",
            "sqft_above": "continuous",
            "sqft_basement": "continuous",
            "yr_built": "continuous",
            "yr_renovated": "continuous",
            "zipcode": "discrete",
            "lat": "continuous",
            "long": "continuous",
            "sqft_living15": "continuous",
            "sqft_lot15": "continuous",
        }

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {
            "sqft_basement": [0.0],
            "yr_renovated": [0.0],
        }
        self.general_columns = [
            "bathrooms",
            "sqft_living",
            "sqft_above",
            "yr_built",
            "long",
            "sqft_living15",
        ]

        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Regression": "price"}


class AbaloneDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/abalone"

    def _get_output(self):
        return "regression"


class BuddyDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/buddy"

    def _get_output(self):
        return "classification"


class CaliforniaDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/california"

    def _get_output(self):
        return "regression"


class CardioDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/cardio"

    def _get_output(self):
        return "classification"


class Churn2Dataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/churn2"

    def _get_output(self):
        return "classification"


class DiabetesOpenMLDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/diabetes_openml"

    def _get_output(self):
        return "classification"


class FBCommentsDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/fb-comments"

    def _get_output(self):
        return "regression"


class GestureDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/gesture"

    def _get_output(self):
        return "classification"


class HiggsSmallDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/higgs-small"

    def _get_output(self):
        return "classification"


class House16HDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/house_16h"

    def _get_output(self):
        return "regression"


class InsuranceDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/insurance"

    def _get_output(self):
        return "regression"


class KingDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/king"

    def _get_output(self):
        return "regression"


class MiniBooneDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/miniboone"

    def _get_output(self):
        return "classification"


class WiltDataset(TabDDPMDataset):
    def _get_base_path(self):
        return "database/dataset/wilt"

    def _get_output(self):
        return "classification"
