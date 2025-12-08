"""
Generative model training algorithm based on the CTABGANSynthesiser

"""

import pandas as pd
import time
from models.CTAB.pipeline.data_preparation import DataPrep
from models.CTAB.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")


class CTABGAN:
    def __init__(
        self,
        args,
        df,
        class_dim=(256, 256, 256, 256),
        random_dim=64,
        num_channels=64,
        batch_size=16000,
        epochs=3000,
        test_ratio=0.20,
        categorical_columns=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
            "income",
        ],
        log_columns=[],
        mixed_columns={"capital-loss": [0.0], "capital-gain": [0.0]},
        general_columns=["age"],
        non_categorical_columns=[],
        integer_columns=[
            "age",
            "fnlwgt",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ],
        problem_type={"Classification": "income"},
        is_loss_corr=0,
        is_loss_dwp=0,
        n_moment_loss_dwp=4,
        checkpoint_freq=50,
    ):
        self.__name__ = "CTABGAN"

        self.synthesizer = CTABGANSynthesizer(
            args=args,
            class_dim=class_dim,
            random_dim=random_dim,
            num_channels=num_channels,
            batch_size=batch_size,
            epochs=epochs,
            checkpoint_freq=checkpoint_freq,
            is_loss_corr=is_loss_corr,
            is_loss_dwp=is_loss_dwp,
            n_moment_loss_dwp=n_moment_loss_dwp,
        )
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

        # Added by the author
        self.df = df
        # Added by the author

    def fit(self, exp_logger):
        start_time = time.time()
        self.data_prep = DataPrep(
            self.df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio,
        )
        self.synthesizer.fit(
            exp_logger=exp_logger,
            data_prep=self.data_prep,
            train_data=self.data_prep.df,
            discrete_columns=self.categorical_columns,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=self.problem_type,
        )
        end_time = time.time()
        print("Finished training in", end_time - start_time, " seconds.")

    def generate_samples(self):
        sample = self.synthesizer.sample(len(self.df))
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
