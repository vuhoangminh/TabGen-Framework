"""Wrapper around CTGAN model."""

import numpy as np
import logging
import rdt
from copy import deepcopy

from models.ctgan import CTGAN

from sdv.errors import NotFittedError
from models.base import BaseSingleTableSynthesizer

from sdv.single_table.copulas import GaussianCopulaSynthesizer
from sdv.single_table.utils import (
    detect_discrete_columns,
    log_numerical_distributions_error,
    validate_numerical_distributions,
)

LOGGER = logging.getLogger(__name__)


class LossValuesMixin:
    """Mixin for accessing loss values from synthesizers."""

    def get_loss_values(self):
        """Get the loss values from the model.

        Raises:
            - ``NotFittedError`` if synthesizer has not been fitted.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.
        """
        if not self._fitted:
            err_msg = (
                "Loss values are not available yet. Please fit your synthesizer first."
            )
            raise NotFittedError(err_msg)

        return self._model.loss_values.copy()


class CTGANSynthesizer(LossValuesMixin, BaseSingleTableSynthesizer):
    """Model wrapping ``CTGAN`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {"categorical": None, "boolean": None}

    def __init__(
        self,
        args,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=2000,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        private=True,
        dp_sigma=1,
        dp_weight_clip=0.01,
        is_loss_corr=0,
        is_loss_dwp=0,
        n_moment_loss_dwp=4,
        is_condvec=1,
        checkpoint_freq=50,
    ):
        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
        )

        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda

        # Added by the author
        self.args = args
        self.private = private
        self.checkpoint_freq = checkpoint_freq
        self.dp_sigma = dp_sigma
        self.dp_weight_clip = dp_weight_clip
        self.is_loss_corr = is_loss_corr
        self.is_loss_dwp = is_loss_dwp
        self.n_moment_loss_dwp = n_moment_loss_dwp
        self.is_condvec = is_condvec
        # Added by the author

        self._model_kwargs = {
            "embedding_dim": embedding_dim,
            "generator_dim": generator_dim,
            "discriminator_dim": discriminator_dim,
            "generator_lr": generator_lr,
            "generator_decay": generator_decay,
            "discriminator_lr": discriminator_lr,
            "discriminator_decay": discriminator_decay,
            "batch_size": batch_size,
            "discriminator_steps": discriminator_steps,
            "log_frequency": log_frequency,
            "verbose": verbose,
            "epochs": epochs,
            "pac": pac,
            "cuda": cuda,
            "args": args,
            "private": private,
            "checkpoint_freq": checkpoint_freq,
            "dp_sigma": dp_sigma,
            "dp_weight_clip": dp_weight_clip,
            "is_loss_corr": is_loss_corr,
            "is_loss_dwp": is_loss_dwp,
            "n_moment_loss_dwp": n_moment_loss_dwp,
            "is_condvec": is_condvec,
        }

    def _estimate_num_columns(self, data):
        """Estimate the number of columns that the data will generate.

        Estimates that continuous columns generate 11 columns and categorical ones
        create n where n is the number of unique categories.

        Args:
            data (pandas.DataFrame):
                Data to estimate the number of columns from.

        Returns:
            int:
                Number of estimate columns.
        """
        sdtypes = self._data_processor.get_sdtypes()
        transformers = self.get_transformers()
        num_generated_columns = {}
        for column in data.columns:
            if column not in sdtypes:
                continue

            if sdtypes[column] in {"numerical", "datetime"}:
                num_generated_columns[column] = 11

            elif sdtypes[column] in {"categorical", "boolean"}:
                if transformers.get(column) is None:
                    num_categories = data[column].fillna(np.nan).nunique(dropna=False)
                    num_generated_columns[column] = num_categories
                else:
                    num_generated_columns[column] = 11

        return num_generated_columns

    def _print_warning(self, data):
        """Print a warning if the number of columns generated is over 1000."""
        dict_generated_columns = self._estimate_num_columns(data)
        if sum(dict_generated_columns.values()) > 1000:
            header = {"Original Column Name  ": "Est # of Columns (CTGAN)"}
            dict_generated_columns = {**header, **dict_generated_columns}
            longest_column_name = len(max(dict_generated_columns, key=len))
            cap = "<" + str(longest_column_name)
            lines_to_print = []
            for column, num_generated_columns in dict_generated_columns.items():
                lines_to_print.append(f"{column:{cap}} {num_generated_columns}")

            generated_columns_str = "\n".join(lines_to_print)
            print(  # noqa: T001
                "PerformanceAlert: Using the CTGANSynthesizer on this data is not recommended. "
                "To model this data, CTGAN will generate a large number of columns."
                "\n\n"
                f"{generated_columns_str}"
                "\n\n"
                "We recommend preprocessing discrete columns that can have many values, "
                "using 'update_transformers'. Or you may drop columns that are not necessary "
                "to model. (Exit this script using ctrl-C)"
            )

    def _preprocess(self, data):
        self.validate(data)
        self._data_processor.fit(data)
        self._print_warning(data)

        return self._data_processor.transform(data)

    def _fit(
        self,
        processed_data,
        exp_logger,
        discrete_columns,
        id_columns=None,
    ):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        self._model = CTGAN(**self._model_kwargs)
        self._model.fit(
            processed_data,
            exp_logger,
            discrete_columns=discrete_columns,
            id_columns=None,
        )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError(
            "CTGANSynthesizer doesn't support conditional sampling."
        )


"""Combination of GaussianCopula transformation and GANs."""


class CopulaGAN(CTGANSynthesizer):
    """Combination of GaussianCopula transformation and GANs.

    This model extends the ``CTGAN`` model to add the flexibility of the GaussianCopula
    transformations provided by the ``GaussianNormalizer`` from ``RDT``.

    Overall, the fitting process consists of the following steps:

    1. Transform each non categorical variable from the input
       data using a ``GaussianNormalizer``:

       i. If not specified, find out the distribution which each one
          of the variables from the input dataset has.
       ii. Transform each variable to a standard normal space by applying
           the CDF of the corresponding distribution and later on applying
           an inverse CDF from a standard normal distribution.

    2. Fit CTGAN with the transformed table.

    And the process of sampling is:

    1. Sample using CTGAN
    2. Reverse the previous transformation by applying the CDF of a standard normal
       distribution and then inverting the CDF of the distribution that correpsonds
       to each variable.

    The arguments of this model are the same as for CTGAN except for two additional
    arguments, ``numerical_distributions`` and ``default_distribution`` that give the
    ability to define specific transformations for individual fields as well as
    which distribution to use by default if no specific distribution has been selected.

    Distributions can be passed as a ``copulas`` univariate instance or as one
    of the following string values:

    * ``norm``: Use a norm distribution.
    * ``beta``: Use a Beta distribution.
    * ``truncnorm``: Use a truncnorm distribution.
    * ``uniform``: Use a uniform distribution.
    * ``gamma``: Use a Gamma distribution.
    * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
      so using this will make ``get_parameters`` unusable.


    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear
            Layer will be created for each one of the values provided. Defaults to (256, 256).
        batch_size (int):
            Number of data samples to process in each step.
        verbose (bool):
            Whether to print fit progress on stdout. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        cuda (bool or str):
            If ``True``, use CUDA. If an ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    _gaussian_normalizer_hyper_transformer = None

    def __init__(
        self,
        args,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        numerical_distributions=None,
        default_distribution=None,
        private=True,
        dp_sigma=1,
        dp_weight_clip=0.01,
        is_loss_corr=0,
        is_loss_dwp=0,
        n_moment_loss_dwp=4,
        is_condvec=1,
        checkpoint_freq=50,
    ):
        super().__init__(
            args,
            metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            epochs=epochs,
            pac=pac,
            cuda=cuda,
            private=private,
            dp_sigma=dp_sigma,
            dp_weight_clip=dp_weight_clip,
            is_loss_corr=is_loss_corr,
            is_loss_dwp=is_loss_dwp,
            n_moment_loss_dwp=n_moment_loss_dwp,
            is_condvec=is_condvec,
            checkpoint_freq=checkpoint_freq,
        )

        validate_numerical_distributions(numerical_distributions, self.metadata.columns)
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution or "beta"

        self._default_distribution = GaussianCopulaSynthesizer.get_distribution_class(
            default_distribution or "beta"
        )
        self._numerical_distributions = {
            field: GaussianCopulaSynthesizer.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }

    def _create_gaussian_normalizer_config(self, processed_data):
        columns = self.metadata.columns
        transformers = {}
        sdtypes = {}
        for column in processed_data.columns:
            sdtype = columns.get(column, {}).get("sdtype")
            if column in columns and sdtype not in ["categorical", "boolean"]:
                sdtypes[column] = "numerical"
                distribution = self._numerical_distributions.get(
                    column, self._default_distribution
                )

                transformers[column] = rdt.transformers.GaussianNormalizer(
                    missing_value_generation="from_column",
                    distribution=distribution,
                )

            else:
                sdtypes[column] = sdtype or "categorical"
                transformers[column] = None

        return {"transformers": transformers, "sdtypes": sdtypes}

    def _fit(
        self,
        processed_data,
        exp_logger,
        discrete_columns,
        id_columns=None,
    ):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        log_numerical_distributions_error(
            self.numerical_distributions, processed_data.columns, LOGGER
        )

        gaussian_normalizer_config = self._create_gaussian_normalizer_config(
            processed_data
        )
        self._gaussian_normalizer_hyper_transformer = rdt.HyperTransformer()
        self._gaussian_normalizer_hyper_transformer.set_config(
            gaussian_normalizer_config
        )
        processed_data = self._gaussian_normalizer_hyper_transformer.fit_transform(
            processed_data
        )

        super()._fit(
            processed_data,
            exp_logger,
            discrete_columns=discrete_columns,
            id_columns=None,
        )

    def fit(
        self,
        data,
        exp_logger,
        discrete_columns,
        id_columns=None,
    ):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        self._fitted = False
        self._data_processor.reset_sampling()
        self._random_state_set = False
        processed_data = self._preprocess(data)

        log_numerical_distributions_error(
            self.numerical_distributions, processed_data.columns, LOGGER
        )

        gaussian_normalizer_config = self._create_gaussian_normalizer_config(
            processed_data
        )
        self._gaussian_normalizer_hyper_transformer = rdt.HyperTransformer()
        self._gaussian_normalizer_hyper_transformer.set_config(
            gaussian_normalizer_config
        )

        processed_data = self._gaussian_normalizer_hyper_transformer.fit_transform(
            processed_data
        )

        # workaround: scipy.stats._warnings_errors.FitError: Optimization converged
        # to parameters that are outside the range allowed by the distribution.
        # -> doesn't work for intrusion
        # try:
        #     processed_data = self._gaussian_normalizer_hyper_transformer.fit_transform(
        #         processed_data
        #     )
        # except:  # intrusion dataset
        #     print(
        #         "failed to _gaussian_normalizer_hyper_transformer, use unprocessed data instead"
        #     )
        #     pass

        self._model = CTGAN(**self._model_kwargs)
        self._model.fit(
            processed_data,
            exp_logger,
            discrete_columns=discrete_columns,
            id_columns=None,
            gaussian_normalizer_hyper_transformer=self._gaussian_normalizer_hyper_transformer,
        )

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = super()._sample(num_rows, conditions)
        return self._gaussian_normalizer_hyper_transformer.reverse_transform(sampled)

    def sample(self, num_rows, conditions=None):
        return self._sample(self, num_rows, conditions)

    def get_learned_distributions(self):
        """Get the marginal distributions used by the ``CTGANSynthesizer``.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        if not self._fitted:
            raise ValueError(
                "Distributions have not been learned yet. Please fit your model first using 'fit'."
            )

        field_transformers = (
            self._gaussian_normalizer_hyper_transformer.field_transformers
        )

        learned_distributions = {}
        for column_name, transformer in field_transformers.items():
            if isinstance(transformer, rdt.transformers.GaussianNormalizer):
                learned_params = deepcopy(transformer._univariate.to_dict())
                learned_params.pop("type")
                distribution = self.numerical_distributions.get(
                    column_name, self.default_distribution
                )
                learned_distributions[column_name] = {
                    "distribution": distribution,
                    "learned_parameters": learned_params,
                }

        return learned_distributions
