import os
import pandas as pd
import argparse

from sdv.metadata import SingleTableMetadata

# from ctgan import CTGAN

from models.ctgan import CTGAN
from models.tvae import TVAE
from models.copulagan import CopulaGAN
from models.dpcgans import DPCGAN
from models.CTAB.ctabgan import CTABGAN

import engine.logger as logger
import engine.utils.model_utils as model_utils
import engine.utils.path_utils as path_utils
import engine.utils.io_utils as io_utils
import engine.utils.print_utils as print_utils
import engine.utils.data_utils as data_utils
from engine.datasets import *


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--dir_logs", type=str, default="database/gan/", help="dir logs")
parser.add_argument(
    "-a",
    "--arch",
    default="ctgan",
    # choices=[
    #     "ctgan",
    #     "tvae",
    #     "copulagan",
    #     "dpcgans",
    #     "ctab",
    #     "all",
    # ],
)
parser.add_argument(
    "--dataset",
    default="adult",
    type=str,
)
parser.add_argument(
    "--epochs",
    type=int,
    # default=3000,
    default=10000,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=16000,
    type=int,
    metavar="N",
)
parser.add_argument(
    "--private",
    default=0,
    type=int,
)
parser.add_argument(
    "-p",
    "--print_freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--is_test",
    default=1,
    type=int,
)
parser.add_argument(
    "--embedding_dim",
    default=128,
    type=int,
)
parser.add_argument(
    "--is_condvec",
    default=1,
    type=int,
)
parser.add_argument(
    "--is_drop_id",
    default=1,
    type=int,
)
parser.add_argument(
    "--is_only_sample",
    default=0,
    type=int,
)
parser.add_argument(
    "--checkpoint_freq",
    default=50,
    type=int,
)
parser.add_argument(
    "--resume",
    default=1,
    type=int,
)

# conventional/proposed method
parser.add_argument(
    "--loss_version",
    default=0,
    choices=[
        0,  # conventional
        2,  # version 2: generalize mean loss to distribution loss and correct correlation loss
        4,  # version 4: only correlation loss
        5,  # version 5: only distribution loss
    ],
    type=int,
)
parser.add_argument(
    "--is_loss_corr",
    default=1,
    type=float,
)
parser.add_argument(
    "--is_loss_dwp",
    default=1,
    type=float,
)
parser.add_argument(
    "--n_moment_loss_dwp",
    default=4,
    type=int,
)


# ctgan
# parser_ctgan = subparsers.add_parser("ctgan")
parser.add_argument(
    "--generator_dim",
    # default="(256,256)",
    default=256,
    type=int,
)
parser.add_argument(
    "--discriminator_dim",
    # default="(256,256)",
    default=256,
    type=int,
)
parser.add_argument(
    "--generator_lr",
    default=2e-4,
    type=float,
)
parser.add_argument(
    "--generator_decay",
    default=1e-6,
    type=float,
)
parser.add_argument(
    "--discriminator_lr",
    default=2e-4,
    type=float,
)
parser.add_argument(
    "--discriminator_decay",
    default=1e-6,
    type=float,
)
parser.add_argument(
    "--discriminator_steps",
    default=1,
    type=int,
)
parser.add_argument(
    "--dp_sigma",
    default=1.0,
    type=float,
)
parser.add_argument(
    "--dp_weight_clip",
    default=0.01,
    type=float,
)

# tvae
# parser_tvae = subparsers.add_parser("tvae")
parser.add_argument(
    "--l2scale",
    default=1e-5,
    type=float,
)
parser.add_argument(
    "--loss_factor",
    default=2,
    type=float,
)
parser.add_argument(
    "--compress_dims",
    default=128,
    type=int,
)
parser.add_argument(
    "--decompress_dims",
    default=128,
    type=int,
)

# ctab
parser.add_argument(
    "--test_ratio",
    default=0.2,
    choices=[0.1, 0.2, 0.3, 0.4, 0.5],
    type=float,
)
parser.add_argument(
    "--n_class_layer",
    default=4,
    choices=[1, 2, 3, 4],
    type=int,
)
parser.add_argument(
    "--class_dim",
    default=32,
    choices=[32, 64, 128, 256],
    type=int,
)
parser.add_argument(
    "--random_dim",
    default=64,
    choices=[16, 32, 64, 128],
    type=int,
)
parser.add_argument(
    "--num_channels",
    default=64,
    choices=[16, 32, 64],
    type=int,
)

# loop train
parser.add_argument(
    "--n_run",
    # default=None,
    default=2,
    type=int,
)

# generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
parser.add_argument(
    "--row_number",
    default=None,
    type=int,
)

# Parse the arguments
args = parser.parse_args()


# ===========================================================================
# main
# ===========================================================================
def main():
    D = get_dataset(args.dataset)

    df = D.data_train
    print(df)

    # sampling rows and columns for the experiment "Evaluating Minimum Data Requirements for Synthetic Data Generation"
    if args.row_number is not None:
        full_df = df.copy()

        subsampled_dfs = data_utils.subsample_dataframe_by_fixed_rows(
            df,
            [args.row_number],
            target_column=D.target,
        )
        df = subsampled_dfs[args.row_number]
        df = df[list(full_df.columns)]

        print(df)

    # Convert all object dtype columns to int
    df = df.apply(
        lambda col: (
            pd.to_numeric(col, errors="ignore") if col.dtype == "object" else col
        )
    )
    if args.row_number is not None:
        full_df = full_df.apply(
            lambda col: (
                pd.to_numeric(col, errors="ignore") if col.dtype == "object" else col
            )
        )
        args.row_number_full = len(full_df)
    else:
        args.row_number_full = None

    if args.is_test and df.shape[0] > 2000:
        df = df.head(2000)

    # get logs directory
    if args.loss_version == 0:
        args.is_loss_corr = 0
        args.is_loss_dwp = 0

    args.dir_logs = os.path.join(
        args.dir_logs, path_utils.get_folder_technical_paper(args)
    )
    print(f">> logging to {args.dir_logs}")

    path_utils.make_dir(args.dir_logs)
    print_utils.print_separator()
    print(f"Save to {args.dir_logs}")
    df.to_csv(
        os.path.join(args.dir_logs, "preprocessed.csv"), sep="\t", encoding="utf-8"
    )

    if args.row_number is not None:
        full_df.to_csv(
            os.path.join(args.dir_logs, "preprocessed_full.csv"),
            sep="\t",
            encoding="utf-8",
        )

    # names of the columns that are discrete
    discrete_columns = D.discrete_columns

    # update discrete_columns based on sampled rows and columns
    discrete_columns = [col for col in discrete_columns if col in df.columns]

    # set up experiment logger
    exp_logger = None

    # set batch_size to df.shape[0] and even
    args.batch_size = min(df.shape[0], args.batch_size)
    # adjust to the nearest number <= args.batch_size that is divisible by 10
    args.batch_size = args.batch_size - (args.batch_size % 100)  # for ctgan pac

    # init model
    if args.arch == "ctgan":
        if exp_logger is None:
            exp_name = os.path.basename(args.dir_logs)  # add timestamp
            exp_logger = logger.Experiment(
                exp_name, io_utils.convert_args_to_dict(args)
            )
            exp_logger.add_meters("train", model_utils.make_meters_ctgan())
        model = CTGAN(
            args,
            embedding_dim=args.embedding_dim,
            generator_dim=io_utils.convert_to_tuple(args.generator_dim, 2),
            discriminator_dim=io_utils.convert_to_tuple(args.discriminator_dim, 2),
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            discriminator_steps=args.discriminator_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            private=args.private,
            dp_sigma=args.dp_sigma,
            dp_weight_clip=args.dp_weight_clip,
            is_loss_corr=args.is_loss_corr,
            is_loss_dwp=args.is_loss_dwp,
            n_moment_loss_dwp=args.n_moment_loss_dwp,
            is_condvec=args.is_condvec,
            checkpoint_freq=args.checkpoint_freq,
            verbose=True,
        )
    elif args.arch == "dpcgans":
        if exp_logger is None:
            exp_name = os.path.basename(args.dir_logs)  # add timestamp
            exp_logger = logger.Experiment(
                exp_name, io_utils.convert_args_to_dict(args)
            )
            exp_logger.add_meters("train", model_utils.make_meters_ctgan())
        model = DPCGAN(
            args,
            embedding_dim=args.embedding_dim,
            generator_dim=io_utils.convert_to_tuple(args.generator_dim),
            discriminator_dim=io_utils.convert_to_tuple(args.discriminator_dim),
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            discriminator_steps=args.discriminator_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            private=args.private,
            dp_sigma=args.dp_sigma,
            dp_weight_clip=args.dp_weight_clip,
            is_loss_corr=args.is_loss_corr,
            is_loss_dwp=args.is_loss_dwp,
            n_moment_loss_dwp=args.n_moment_loss_dwp,
            is_condvec=args.is_condvec,
            checkpoint_freq=args.checkpoint_freq,
            verbose=True,
        )
    elif args.arch == "copulagan":
        if exp_logger is None:
            exp_name = os.path.basename(args.dir_logs)  # add timestamp
            exp_logger = logger.Experiment(
                exp_name, io_utils.convert_args_to_dict(args)
            )
            exp_logger.add_meters("train", model_utils.make_meters_ctgan())
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # fix bug metadata
        if args.dataset == "house":
            metadata.columns["sqft_living"] = {"sdtype": "numerical"}
            metadata.columns["sqft_living15"] = {"sdtype": "numerical"}
            metadata.columns["zipcode"] = {"sdtype": "categorical"}

        model = CopulaGAN(
            args,
            metadata=metadata,
            embedding_dim=args.embedding_dim,
            generator_dim=io_utils.convert_to_tuple(args.generator_dim),
            discriminator_dim=io_utils.convert_to_tuple(args.discriminator_dim),
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            discriminator_steps=args.discriminator_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            private=args.private,
            dp_sigma=args.dp_sigma,
            dp_weight_clip=args.dp_weight_clip,
            is_loss_corr=args.is_loss_corr,
            is_loss_dwp=args.is_loss_dwp,
            n_moment_loss_dwp=args.n_moment_loss_dwp,
            is_condvec=args.is_condvec,
            checkpoint_freq=args.checkpoint_freq,
            verbose=True,
        )
    elif args.arch == "tvae":
        if exp_logger is None:
            exp_name = os.path.basename(args.dir_logs)  # add timestamp
            exp_logger = logger.Experiment(
                exp_name, io_utils.convert_args_to_dict(args)
            )
            exp_logger.add_meters("train", model_utils.make_meters_tvae())
        model = TVAE(
            args,
            embedding_dim=args.embedding_dim,
            compress_dims=io_utils.convert_to_tuple(args.compress_dims),
            decompress_dims=io_utils.convert_to_tuple(args.decompress_dims),
            l2scale=args.l2scale,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_factor=args.loss_factor,
            is_loss_corr=args.is_loss_corr,
            is_loss_dwp=args.is_loss_dwp,
            n_moment_loss_dwp=args.n_moment_loss_dwp,
            checkpoint_freq=args.checkpoint_freq,
            verbose=True,
        )
    elif args.arch == "ctab":
        if exp_logger is None:
            exp_name = os.path.basename(args.dir_logs)  # add timestamp
            exp_logger = logger.Experiment(
                exp_name, io_utils.convert_args_to_dict(args)
            )
            exp_logger.add_meters("train", model_utils.make_meters_ctgan())
        model = CTABGAN(
            args=args,
            df=df,
            test_ratio=args.test_ratio,
            class_dim=io_utils.convert_to_tuple(args.class_dim, n=args.n_class_layer),
            random_dim=args.random_dim,
            num_channels=args.num_channels,
            categorical_columns=D.categorical_columns,
            log_columns=D.log_columns,
            mixed_columns=D.mixed_columns,
            general_columns=D.general_columns,
            non_categorical_columns=D.non_categorical_columns,
            integer_columns=D.integer_columns,
            problem_type=D.problem_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            is_loss_corr=args.is_loss_corr,
            is_loss_dwp=args.is_loss_dwp,
            n_moment_loss_dwp=args.n_moment_loss_dwp,
            checkpoint_freq=args.checkpoint_freq,
        )
    else:
        raise NotImplementedError

    if args.arch in ["ctgan", "tvae", "copulagan", "dpcgans"]:
        model.fit(
            df,
            exp_logger,
            discrete_columns=discrete_columns,
            id_columns=None,
        )
    elif args.arch in ["ctab"]:
        model.fit(exp_logger)

    return args.dir_logs


if __name__ == "__main__":
    main()
