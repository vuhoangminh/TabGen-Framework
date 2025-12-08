"""CTGAN cusctom module."""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import time
from torch import optim
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    functional,
    InstanceNorm1d,
)

# from ctgan.data_sampler import DataSampler
from engine.ctgan_data_sampler import MyDataset, MyCustomDataset
from engine.ctgan_data_transformer import DataTransformer

from ctgan.synthesizers.base import BaseSynthesizer, random_state
from engine.rdp_accountant import compute_rdp, get_privacy_spent

from torch.utils.data import DataLoader
from engine.utils.model_utils import save_checkpoint
import engine.utils.train_utils as train_utils
import engine.utils.print_utils as print_utils
import engine.utils.model_utils as model_utils
import engine.utils.eval_utils as eval_utils
import engine.utils.path_utils as path_utils
from engine.config import config

from engine.custom_loss import (
    CorrectedCorrelationLoss,
    DistributionLoss,
)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=10, lambda_=10
    ):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)

        # Added by the author
        self.bn = InstanceNorm1d(o)
        # self.bn = BatchNorm1d(o)
        # Added by the author

        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
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
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        args,
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
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

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

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(
                logits, tau=tau, hard=hard, eps=eps, dim=dim
            )
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError("gumbel_softmax returning NaN.")

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(
                        f"Unexpected activation function {span_info.activation_fn}."
                    )

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    @random_state
    def fit(
        self,
        train_data,
        exp_logger,
        discrete_columns=(),
        id_columns=(),
        epochs=None,
        experiment=None,
        gaussian_normalizer_hyper_transformer=None,
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        # # Added by the author -- id
        # self._validate_discrete_columns(train_data, id_columns)
        # # Added by the author -- id

        # Added by the author -- corr + dwp
        self._corrcoef_original = torch.corrcoef(
            torch.transpose(
                torch.tensor(train_data.to_numpy(), dtype=torch.float32), 0, 1
            ).to(self._device)
        )
        self._corrcoef_original = torch.nan_to_num(self._corrcoef_original, nan=0)
        self.df = train_data.copy()
        # Added by the author -- corr + dwp

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        print(">> transforming data")
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        # Added by the author -- opacus
        self._data_sampler = MyDataset(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        # Added by the author -- opacus

        data_dim = self._transformer.output_dimensions

        # Added by the author -- condvec
        if self.is_condvec:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
            ).to(self._device)
        else:
            self._generator = Generator(
                self._embedding_dim,
                self._generator_dim,
                data_dim,
            ).to(self._device)

        if self.is_condvec:
            discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vec(),
                self._discriminator_dim,
                pac=self.pac,
            ).to(self._device)
        else:
            discriminator = Discriminator(
                data_dim,
                self._discriminator_dim,
                pac=self.pac,
            ).to(self._device)
        # Added by the author -- condvec

        # Added by the author -- logger
        exp_logger.info["generator_model_params"] = train_utils.params_count(
            self._generator
        )
        exp_logger.info["discriminator_model_params"] = train_utils.params_count(
            discriminator
        )
        # Added by the author -- logger

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        # Added by the author -- load
        if self.args.resume:
            try:
                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    self._generator, optimizerG, self.args.dir_logs, suffix="generator"
                )

                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    discriminator,
                    optimizerD,
                    self.args.dir_logs,
                    suffix="discriminator",
                )
            except:
                print("Something is wrong. Can't load last model")
                self.args.start_epoch = 0
        # Added by the author -- load

        """
        # Added by the author -- opacus
        self._dataloader = MyDataLoader(self._data_sampler, batch_size=self._batch_size)
        if True:
            privacy_engine = PrivacyEngine(secure_mode=False)

            self._generator, optimizerG, self._dataloader = privacy_engine.make_private(
                module=self._generator,
                optimizer=optimizerG,
                data_loader=self._dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
        # dataiter = iter(self._dataloader)
        # Added by the author -- opacus
        """

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        # Added by the author -- logger
        meters = exp_logger.reset_meters("train")
        start_epoch = time.time()
        # Added by the author -- logger

        # Added by the author -- DP
        # we moved it out of the loop because  we want to register it only once
        # for the whole training. Otherwise we add a new hook everytime and
        # we will have many hooks which makes epoch time increase over epoch
        if self.private:
            for parameter in discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad.cuda()
                    + (1 / self._batch_size)
                    * self.dp_sigma
                    * torch.randn(parameter.shape).cuda()
                )
        # Added by the author -- DP

        train_data_tensor = torch.tensor(train_data).to(self._device)

        print(">> starting traning")
        for i_epoch in range(epochs):
            # Print the maximum and minimum values of all parameters -- debug purpose
            all_params = torch.cat([p.view(-1) for p in discriminator.parameters()])
            # print("Max value:", all_params.max().item())
            # print("Min value:", all_params.min().item())

            if self.args.resume and i_epoch < self.args.start_epoch:
                continue
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    # Added by the author -- condvec
                    if not self.is_condvec or condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )
                    # Added by the author -- condvec
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    # Added by the author -- condvec
                    if not self.is_condvec or c1 is None:
                        real_cat = real
                        fake_cat = fakeact
                    else:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    # Added by the author -- condvec

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    # Added by the author -- logger
                    meters["loss_d"].update(loss_d.item(), fake.size(0))
                    # Added by the author -- logger

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    # Added by the author -- DP
                    if self.private:
                        # Weight clipping for privacy guarantee
                        for param in discriminator.parameters():
                            param.data.clamp_(
                                -self.dp_weight_clip,
                                self.dp_weight_clip,
                            )
                    # Added by the author -- DP

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                # Added by the author -- condvec
                if not self.is_condvec or condvec is None:
                    c1, m1, col, opt = None, None, None, None
                # Added by the author -- condvec
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # Added by the author -- condvec
                if not self.is_condvec or c1 is None:
                    y_fake = discriminator(fakeact)
                else:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))

                if not self.is_condvec or condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                # Added by the author -- condvec

                # Added by the author -- corr + dwp
                df_fake_inverse = self._transformer.inverse_transform(
                    fakeact.detach().cpu().numpy()
                )
                if (
                    self.args.loss_version > 0
                    and self.is_loss_corr
                    # Added by the author -- only dis. loss
                    and self.args.loss_version != 5
                    # Added by the author -- only dis. loss
                ):
                    loss_corr = self.is_loss_corr * CorrectedCorrelationLoss()(
                        train_data_tensor, fakeact
                    )  # version 2 and 3 and 4
                    meters["loss_corr"].update(loss_corr.item(), fake.size(0))
                else:
                    loss_corr = 0
                    meters["loss_corr"].update(0, fake.size(0))

                if (
                    self.args.loss_version > 0
                    and self.is_loss_dwp
                    # Added by the author -- only cor. loss
                    and self.args.loss_version != 4
                    # Added by the author -- only cor. loss
                ):
                    loss_dwp = self.is_loss_dwp * DistributionLoss()(
                        train_data_tensor,
                        fakeact,
                        alpha=1,
                        n=self.n_moment_loss_dwp,
                    )  # version 2 and 5
                    meters["loss_dwp"].update(loss_dwp.item(), fake.size(0))
                else:
                    loss_dwp = 0
                    meters["loss_dwp"].update(0, fake.size(0))

                metric_corr = eval_utils.compute_diff_correlation(
                    self.df, df_fake_inverse
                )
                meters["metric_corr"].update(metric_corr, 1)
                metric_dwp, _, _ = eval_utils.compute_dwp(
                    self.df, df_fake_inverse, discrete_columns=discrete_columns
                )
                meters["metric_dwp"].update(metric_dwp, 1)

                # total loss
                loss_g = -torch.mean(y_fake) + cross_entropy + loss_corr + loss_dwp
                # Added by the author -- corr + dwp

                # Added by the author -- logger
                meters["loss_g"].update(loss_g.item(), fake.size(0))
                # Added by the author -- logger

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            # measure elapsed time
            meters["epoch_time"].update(time.time() - start_epoch)
            start_epoch = time.time()

            if self._verbose:
                curr_epoch = i_epoch + 1
                txt_epoch = "{:<10s}".format("Epoch:") + "{curr_epoch:04}".format(
                    curr_epoch=curr_epoch
                )
                txt_epoch_time = "{:<8s}".format(
                    "Etime:"
                ) + "{epoch_time.val:.4f}".format(epoch_time=meters["epoch_time"])
                txt_g = "{:<8s}".format("LossG:") + "{loss_g.avg:.4f}".format(
                    loss_g=meters["loss_g"]
                )
                txt_d = "{:<8s}".format("LossD:") + "{loss_d.avg:.4f}".format(
                    loss_d=meters["loss_d"]
                )
                txt_corr = "{:<8s}".format("LossC:") + "{loss_corr.avg:.4f}".format(
                    loss_corr=meters["loss_corr"]
                )
                txt_dwp = "{:<8s}".format("LossP:") + "{loss_dwp.avg:.4f}".format(
                    loss_dwp=meters["loss_dwp"]
                )

                if meters["loss_d"].avg < 0:
                    txt_metric_corr = "{:<9s}".format(
                        "MCorr:"
                    ) + "{metric_corr.avg:.4f}".format(
                        metric_corr=meters["metric_corr"]
                    )
                else:
                    txt_metric_corr = "{:<8s}".format(
                        "MCorr:"
                    ) + "{metric_corr.avg:.4f}".format(
                        metric_corr=meters["metric_corr"]
                    )

                if meters["loss_g"].avg < 0:
                    txt_metric_dwp = "{:<9s}".format(
                        "MDwp:"
                    ) + "{metric_dwp.avg:.4f}".format(metric_dwp=meters["metric_dwp"])
                else:
                    txt_metric_dwp = "{:<8s}".format(
                        "MDwp:"
                    ) + "{metric_dwp.avg:.4f}".format(metric_dwp=meters["metric_dwp"])

                print()
                print()
                print_utils.print_separator()
                print(
                    f"{txt_epoch:<25s} {txt_epoch_time:<25s} {txt_d:<25s} {txt_g:<25s}"
                )
                print(
                    f"{txt_corr:<25s} {txt_dwp:<25s} {txt_metric_corr:<25s} {txt_metric_dwp:<25s}"
                )

                # print_utils.print_separator()

                # Added by the author -- DP
                if self.private:
                    orders = [1 + x / 10.0 for x in range(1, 100)]
                    sampling_probability = self._batch_size / len(train_data)
                    delta = 2e-6
                    rdp = compute_rdp(
                        q=sampling_probability,
                        noise_multiplier=self.dp_sigma,
                        steps=i_epoch * steps_per_epoch,
                        orders=orders,
                    )
                    epsilon, _, opt_order = get_privacy_spent(
                        orders, rdp, target_delta=delta
                    )  # target_delta=1e-5

                    print_utils.print_separator()
                    print(
                        "differential privacy with eps = {:.3g} and delta = {}.".format(
                            epsilon, delta
                        )
                    )
                    print("The optimal RDP order is {}.".format(opt_order))

                    if opt_order == max(orders) or opt_order == min(orders):
                        print(
                            "The privacy estimate is likely to be improved by expanding the set of orders."
                        )

                    # print_utils.print_separator()

                    meters["dp_sigma"].update(self.dp_sigma)
                    meters["dp_weight_clip"].update(self.dp_weight_clip)
                    meters["dp_epsilon"].update(epsilon)
                    meters["dp_delta"].update(delta)
                    meters["dp_opt_order"].update(opt_order)
                else:
                    epsilon = np.nan
                # Added by the author -- DP

            # Added by the author -- save model + sample
            # if (
            #     self.checkpoint_freq is not None
            #     and (i_epoch + 1) % self.checkpoint_freq == 0
            # ):
            if i_epoch + 1 == epochs:  # save storage
                print_utils.print_processing("save models")
                save_checkpoint(
                    {
                        "epoch": i_epoch + 1,
                        "arch": self.args.arch,
                        "best_metric": 0,
                        "exp_logger": exp_logger,
                    },
                    self._generator.state_dict(),
                    optimizerG.state_dict(),
                    self.args.dir_logs,
                    i_epoch=i_epoch + 1,
                    save_model=True,
                    save_all_from=None,
                    is_best=True,
                    suffix="generator",
                )

                save_checkpoint(
                    {
                        "epoch": i_epoch + 1,
                        "arch": self.args.arch,
                        "best_metric": 0,
                        "exp_logger": exp_logger,
                    },
                    discriminator.state_dict(),
                    optimizerD.state_dict(),
                    self.args.dir_logs,
                    i_epoch=i_epoch + 1,
                    save_model=True,
                    save_all_from=None,
                    is_best=True,
                    suffix="discriminator",
                )

                print_utils.print_processing("generate synthetic data")
                n_samples = int(len(train_data) * 1.05)
                synthetic_data = self.sample(n_samples)
                # copulagan
                if gaussian_normalizer_hyper_transformer is not None:
                    synthetic_data = (
                        gaussian_normalizer_hyper_transformer.reverse_transform(
                            synthetic_data
                        )
                    )
                synthetic_data.to_csv(
                    os.path.join(self.args.dir_logs, f"fake_{i_epoch+1:05}.csv"),
                    sep="\t",
                    encoding="utf-8",
                )

            exp_logger.log_meters("train", n=i_epoch)
            exp_logger.to_json(
                os.path.join(self.args.dir_logs, "logger.json")
            )  # overwrite logger.json after each epoch
            # Added by the author -- save model + sample

        if self.args.row_number_full is not None:
            print_utils.print_processing("generate full synthetic data")
            n_samples = int(self.args.row_number_full * 1.2)
            synthetic_data = self.sample(n_samples)
            # copulagan
            if gaussian_normalizer_hyper_transformer is not None:
                synthetic_data = (
                    gaussian_normalizer_hyper_transformer.reverse_transform(
                        synthetic_data
                    )
                )
            synthetic_data.to_csv(
                os.path.join(self.args.dir_logs, f"synthetic_full.csv"),
                sep="\t",
                encoding="utf-8",
            )

    @random_state
    def load_and_sample(
        self,
        train_data,
        exp_logger,
        discrete_columns=(),
        epochs=None,
        is_exluded_real_data=True,
        n_sample=500_000,
    ):
        """Load the CTGAN Synthesizer models to sample.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        df = train_data.copy()

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        # Added by the author -- opacus
        self._data_sampler = MyDataset(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        # Added by the author -- opacus

        data_dim = self._transformer.output_dimensions

        # Added by the author -- condvec
        if self.is_condvec:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
            ).to(self._device)
        else:
            self._generator = Generator(
                self._embedding_dim,
                self._generator_dim,
                data_dim,
            ).to(self._device)
        # Added by the author -- condvec

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        # Added by the author -- load
        (
            self.args.start_epoch,
            best_metric,
            exp_logger,
            is_loadable_resume,
        ) = model_utils.load_checkpoint(
            self._generator, optimizerG, self.args.dir_logs, suffix="generator"
        )
        # Added by the author -- load

        if not is_loadable_resume:
            raise LookupError("Failed to load model")
        # Added by the author -- load

        # We generate until synthetic data have all classes in the real data
        folder_dir = "database/synthetic"
        path_utils.make_dir(folder_dir)
        filename = path_utils.get_filename(self.args.dir_logs)
        file_path = os.path.join(folder_dir, f"{filename}.csv")

        print(f">> sampling and saving to {file_path}")

        is_sample = True
        while is_sample:
            print_utils.print_separator()
            synthetic_data = self.sample(n_sample)
            is_sample_have_all_classes = eval_utils.is_sample_have_all_classes(
                df, synthetic_data
            )
            if is_sample_have_all_classes:
                is_sample = False
        # We generate until synthetic data have all classes in the real data

        synthetic_data.to_csv(
            file_path,
            sep="\t",
            encoding="utf-8",
        )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            # Added by the author -- condvec
            elif self.is_condvec:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)
            else:
                condvec = None
            # Added by the author -- condvec

            # Added by the author -- condvec
            if not self.is_condvec or condvec is None:
                pass
            # Added by the author -- condvec
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    # bug with dataloader when last batch is incomplete
    @random_state
    def _legacy_fit(
        self, train_data, exp_logger, discrete_columns=(), epochs=None, experiment=None
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        # Added by the author -- corr + dwp
        self._corrcoef_original = torch.corrcoef(
            torch.transpose(
                torch.tensor(train_data.to_numpy(), dtype=torch.float32), 0, 1
            ).to(self._device)
        )
        self._corrcoef_original = torch.nan_to_num(self._corrcoef_original, nan=0)
        self.df = train_data.copy()
        # Added by the author -- corr + dwp

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        # Added by the author -- opacus
        self._data_sampler = MyDataset(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        # Added by the author -- opacus

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        # Added by the author -- logger
        exp_logger.info["generator_model_params"] = train_utils.params_count(
            self._generator
        )
        exp_logger.info["discriminator_model_params"] = train_utils.params_count(
            discriminator
        )
        # Added by the author -- logger

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        # Added by the author -- load
        if self.args.resume:
            try:
                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    self._generator, optimizerG, self.args.dir_logs, suffix="generator"
                )

                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    discriminator,
                    optimizerD,
                    self.args.dir_logs,
                    suffix="discriminator",
                )
            except:
                print("Something is wrong. Can't load last model")
                self.args.start_epoch = 1

        # Added by the author -- load

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        # Added by the author -- logger
        meters = exp_logger.reset_meters("train")
        start_epoch = time.time()
        # Added by the author -- logger

        # Added by the author -- DP
        # we moved it out of the loop because  we want to register it only once
        # for the whole training. Otherwise we add a new hook everytime and
        # we will have many hooks which makes epoch time increase over epoch
        if self.private:
            for parameter in discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad.cuda()
                    + (1 / self._batch_size)
                    * self.dp_sigma
                    * torch.randn(parameter.shape).cuda()
                )
        # Added by the author -- DP

        for i_epoch in range(epochs):
            if self.args.resume and i_epoch < self.args.start_epoch:
                continue
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    # Added by the author -- condvec
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )
                    # Added by the author -- condvec
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    # Added by the author -- condvec
                    if c1 is None:
                        real_cat = real
                        fake_cat = fakeact
                    else:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    # Added by the author -- condvec

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    # Added by the author -- logger
                    meters["loss_d"].update(loss_d.item(), fake.size(0))
                    # Added by the author -- logger

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    # Added by the author -- DP
                    if self.private:
                        # Weight clipping for privacy guarantee
                        for param in discriminator.parameters():
                            param.data.clamp_(
                                -self.dp_weight_clip,
                                self.dp_weight_clip,
                            )
                    # Added by the author -- DP

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                # Added by the author -- condvec
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                # Added by the author -- condvec
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # Added by the author -- condvec
                if c1 is None:
                    y_fake = discriminator(fakeact)
                else:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                # Added by the author -- condvec

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                # Added by the author -- corr + dwp
                if self.is_loss_dwp or self.is_loss_corr:
                    df_fake_inverse = self._transformer.inverse_transform(
                        fakeact.detach().cpu().numpy()
                    )
                if self.args.loss_version > 0 and self.is_loss_corr:
                    _corrcoef_fake = torch.corrcoef(
                        torch.transpose(
                            torch.tensor(
                                df_fake_inverse.to_numpy(),
                                dtype=torch.float32,
                            ),
                            0,
                            1,
                        ).to(self._device)
                    )
                    _corrcoef_fake = torch.nan_to_num(_corrcoef_fake, nan=0)
                    loss_corr = (
                        torch.sum(
                            torch.absolute(_corrcoef_fake - self._corrcoef_original)
                        )
                        / torch.numel(_corrcoef_fake)
                        * 10
                    )
                    meters["loss_corr"].update(loss_corr.item(), fake.size(0))
                else:
                    loss_corr = 0
                    meters["loss_corr"].update(0, fake.size(0))

                if self.args.loss_version > 0 and self.is_loss_dwp:
                    loss_dwp = torch.tensor(
                        eval_utils.compute_dwp(self.df, df_fake_inverse)[0] * 100
                    ).to(self._device)
                    meters["loss_dwp"].update(loss_dwp.item(), fake.size(0))
                else:
                    loss_dwp = 0
                    meters["loss_dwp"].update(0, fake.size(0))

                loss_g = -torch.mean(y_fake) + cross_entropy + loss_corr + loss_dwp
                # Added by the author -- corr + dwp

                # Added by the author -- logger
                meters["loss_g"].update(loss_g.item(), fake.size(0))
                # Added by the author -- logger

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            # measure elapsed time
            meters["epoch_time"].update(time.time() - start_epoch)
            start_epoch = time.time()

            if self._verbose:
                print()
                print_utils.print_separator()
                txt_epoch = f"Epoch: [{i_epoch:04}/{epochs:04}]".format(
                    i_epoch=i_epoch + 1,
                    epochs=epochs,
                )
                txt_epoch_time = "Epoch_time: {epoch_time.val:.2f}".format(
                    epoch_time=meters["epoch_time"]
                )
                txt_g = "Loss_g: {loss_g.val:.4f} ({loss_g.avg:.4f})".format(
                    loss_g=meters["loss_g"]
                )
                txt_d = "Loss_d: {loss_d.val:.4f} ({loss_d.avg:.4f})".format(
                    loss_d=meters["loss_d"]
                )
                txt_corr = (
                    "Loss_corr: {loss_corr.val:.4f} ({loss_corr.avg:.4f})".format(
                        loss_corr=meters["loss_corr"]
                    )
                )
                txt_dwp = "Loss_dwp: {loss_dwp.val:.4f} ({loss_dwp.avg:.4f})".format(
                    loss_dwp=meters["loss_dwp"]
                )

                print(
                    "{:<15s}     {:<15s}     {:<22s}     {:<22s}     {:<25s}     {:<25s}".format(
                        txt_epoch, txt_epoch_time, txt_d, txt_g, txt_corr, txt_dwp
                    )
                )
                print_utils.print_separator()

                # Added by the author -- DP
                if self.private:
                    orders = [1 + x / 10.0 for x in range(1, 100)]
                    sampling_probability = self._batch_size / len(train_data)
                    delta = 2e-6
                    rdp = compute_rdp(
                        q=sampling_probability,
                        noise_multiplier=self.dp_sigma,
                        steps=i_epoch * steps_per_epoch,
                        orders=orders,
                    )
                    epsilon, _, opt_order = get_privacy_spent(
                        orders, rdp, target_delta=delta
                    )  # target_delta=1e-5

                    print(
                        "differential privacy with eps = {:.3g} and delta = {}.".format(
                            epsilon, delta
                        )
                    )
                    print("The optimal RDP order is {}.".format(opt_order))

                    if opt_order == max(orders) or opt_order == min(orders):
                        print(
                            "The privacy estimate is likely to be improved by expanding the set of orders."
                        )

                    print_utils.print_separator()

                    meters["dp_sigma"].update(self.dp_sigma)
                    meters["dp_weight_clip"].update(self.dp_weight_clip)
                    meters["dp_epsilon"].update(epsilon)
                    meters["dp_delta"].update(delta)
                    meters["dp_opt_order"].update(opt_order)
                else:
                    epsilon = np.nan
                # Added by the author -- DP

            # Added by the author -- save model + sample
            if self.checkpoint_freq is not None:
                if (i_epoch + 1) % self.checkpoint_freq == 0:
                    save_checkpoint(
                        {
                            "epoch": i_epoch + 1,
                            "arch": self.args.arch,
                            "best_metric": 0,
                            "exp_logger": exp_logger,
                        },
                        self._generator.state_dict(),
                        optimizerG.state_dict(),
                        self.args.dir_logs,
                        i_epoch=i_epoch + 1,
                        save_model=True,
                        save_all_from=None,
                        is_best=True,
                        suffix="generator",
                    )

                    save_checkpoint(
                        {
                            "epoch": i_epoch + 1,
                            "arch": self.args.arch,
                            "best_metric": 0,
                            "exp_logger": exp_logger,
                        },
                        discriminator.state_dict(),
                        optimizerD.state_dict(),
                        self.args.dir_logs,
                        i_epoch=i_epoch + 1,
                        save_model=True,
                        save_all_from=None,
                        is_best=True,
                        suffix="discriminator",
                    )

                    n_samples = int(len(train_data) * 1.05)
                    synthetic_data = self.sample(n_samples)
                    synthetic_data.to_csv(
                        os.path.join(self.args.dir_logs, f"fake_{i_epoch+1:05}.csv"),
                        sep="\t",
                        encoding="utf-8",
                    )

            exp_logger.log_meters("train", n=i_epoch)
            exp_logger.to_json(
                os.path.join(self.args.dir_logs, "logger.json")
            )  # overwrite logger.json after each epoch
            # Added by the author -- save model + sample

    # bug with dataloader when last batch is incomplete
    @random_state
    def _legacy_fit_no_condvec(
        self, train_data, exp_logger, discrete_columns=(), epochs=None, experiment=None
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        # Added by the author -- corr + dwp
        self._corrcoef_original = torch.corrcoef(
            torch.transpose(
                torch.tensor(train_data.to_numpy(), dtype=torch.float32), 0, 1
            ).to(self._device)
        )
        self._corrcoef_original = torch.nan_to_num(self._corrcoef_original, nan=0)
        self.df = train_data.copy()
        # Added by the author -- corr + dwp

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        # Added by the author -- opacus
        self._data_sampler = DataLoader(
            MyCustomDataset(train_data),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
        )
        # Added by the author -- opacus

        data_dim = self._transformer.output_dimensions

        # Added by the author -- condvec
        self._generator = Generator(
            self._embedding_dim,
            self._generator_dim,
            data_dim,
        ).to(self._device)
        discriminator = Discriminator(
            data_dim,
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)
        # Added by the author -- condvec

        # Added by the author -- logger
        exp_logger.info["generator_model_params"] = train_utils.params_count(
            self._generator
        )
        exp_logger.info["discriminator_model_params"] = train_utils.params_count(
            discriminator
        )
        # Added by the author -- logger

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        # Added by the author -- load
        if self.args.resume:
            try:
                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    self._generator, optimizerG, self.args.dir_logs, suffix="generator"
                )

                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    discriminator,
                    optimizerD,
                    self.args.dir_logs,
                    suffix="discriminator",
                )
            except:
                print("Something is wrong. Can't load last model")
                self.args.start_epoch = 1
        # Added by the author -- load

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        # Added by the author -- logger
        meters = exp_logger.reset_meters("train")
        start_epoch = time.time()
        # Added by the author -- logger

        # Added by the author -- DP
        # we moved it out of the loop because  we want to register it only once
        # for the whole training. Otherwise we add a new hook everytime and
        # we will have many hooks which makes epoch time increase over epoch
        if self.private:
            for parameter in discriminator.parameters():
                parameter.register_hook(
                    lambda grad: grad.cuda()
                    + (1 / self._batch_size)
                    * self.dp_sigma
                    * torch.randn(parameter.shape).cuda()
                )
        # Added by the author -- DP

        for i_epoch in range(epochs):
            if self.args.resume and i_epoch < self.args.start_epoch:
                continue
            for i_batch, real in enumerate(self._data_sampler):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.numpy().astype("float32")).to(
                        self._device
                    )

                    # Added by the author -- condvec
                    real_cat = real
                    fake_cat = fakeact
                    # Added by the author -- condvec

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    # Added by the author -- logger
                    meters["loss_d"].update(loss_d.item(), fake.size(0))
                    # Added by the author -- logger

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                    # Added by the author -- DP
                    if self.private:
                        # Weight clipping for privacy guarantee
                        for param in discriminator.parameters():
                            param.data.clamp_(
                                -self.dp_weight_clip,
                                self.dp_weight_clip,
                            )
                    # Added by the author -- DP

                fakez = torch.normal(mean=mean, std=std)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                # Added by the author -- condvec
                y_fake = discriminator(fakeact)
                # Added by the author -- condvec

                cross_entropy = 0

                # Added by the author -- corr + dwp
                if self.is_loss_dwp or self.is_loss_corr:
                    df_fake_inverse = self._transformer.inverse_transform(
                        fakeact.detach().cpu().numpy()
                    )
                if self.args.loss_version > 0 and self.is_loss_corr:
                    _corrcoef_fake = torch.corrcoef(
                        torch.transpose(
                            torch.tensor(
                                df_fake_inverse.to_numpy(),
                                dtype=torch.float32,
                            ),
                            0,
                            1,
                        ).to(self._device)
                    )
                    _corrcoef_fake = torch.nan_to_num(_corrcoef_fake, nan=0)
                    loss_corr = (
                        torch.sum(
                            torch.absolute(_corrcoef_fake - self._corrcoef_original)
                        )
                        / torch.numel(_corrcoef_fake)
                        * 10
                    )
                    meters["loss_corr"].update(loss_corr.item(), fake.size(0))
                else:
                    loss_corr = 0
                    meters["loss_corr"].update(0, fake.size(0))

                if self.args.loss_version > 0 and self.is_loss_dwp:
                    loss_dwp = torch.tensor(
                        eval_utils.compute_dwp(self.df, df_fake_inverse)[0] * 100
                    ).to(self._device)
                    meters["loss_dwp"].update(loss_dwp.item(), fake.size(0))
                else:
                    loss_dwp = 0
                    meters["loss_dwp"].update(0, fake.size(0))

                loss_g = -torch.mean(y_fake) + cross_entropy + loss_corr + loss_dwp
                # Added by the author -- corr + dwp

                # Added by the author -- logger
                meters["loss_g"].update(loss_g.item(), fake.size(0))
                # Added by the author -- logger

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            # measure elapsed time
            meters["epoch_time"].update(time.time() - start_epoch)
            start_epoch = time.time()

            if self._verbose:
                print()
                print_utils.print_separator()
                txt_epoch = f"Epoch: [{i_epoch:04}/{epochs:04}]".format(
                    i_epoch=i_epoch + 1,
                    epochs=epochs,
                )
                txt_epoch_time = "Epoch_time: {epoch_time.val:.2f}".format(
                    epoch_time=meters["epoch_time"]
                )
                txt_g = "Loss_g: {loss_g.val:.4f} ({loss_g.avg:.4f})".format(
                    loss_g=meters["loss_g"]
                )
                txt_d = "Loss_d: {loss_d.val:.4f} ({loss_d.avg:.4f})".format(
                    loss_d=meters["loss_d"]
                )
                txt_corr = (
                    "Loss_corr: {loss_corr.val:.4f} ({loss_corr.avg:.4f})".format(
                        loss_corr=meters["loss_corr"]
                    )
                )
                txt_dwp = "Loss_dwp: {loss_dwp.val:.4f} ({loss_dwp.avg:.4f})".format(
                    loss_dwp=meters["loss_dwp"]
                )

                print(
                    "{:<15s}     {:<15s}     {:<22s}     {:<22s}     {:<25s}     {:<25s}".format(
                        txt_epoch, txt_epoch_time, txt_d, txt_g, txt_corr, txt_dwp
                    )
                )
                print_utils.print_separator()

                # Added by the author -- DP
                if self.private:
                    orders = [1 + x / 10.0 for x in range(1, 100)]
                    sampling_probability = self._batch_size / len(train_data)
                    delta = 2e-6
                    rdp = compute_rdp(
                        q=sampling_probability,
                        noise_multiplier=self.dp_sigma,
                        steps=i_epoch * steps_per_epoch,
                        orders=orders,
                    )
                    epsilon, _, opt_order = get_privacy_spent(
                        orders, rdp, target_delta=delta
                    )  # target_delta=1e-5

                    print(
                        "differential privacy with eps = {:.3g} and delta = {}.".format(
                            epsilon, delta
                        )
                    )
                    print("The optimal RDP order is {}.".format(opt_order))

                    if opt_order == max(orders) or opt_order == min(orders):
                        print(
                            "The privacy estimate is likely to be improved by expanding the set of orders."
                        )

                    print_utils.print_separator()

                    meters["dp_sigma"].update(self.dp_sigma)
                    meters["dp_weight_clip"].update(self.dp_weight_clip)
                    meters["dp_epsilon"].update(epsilon)
                    meters["dp_delta"].update(delta)
                    meters["dp_opt_order"].update(opt_order)
                else:
                    epsilon = np.nan
                # Added by the author -- DP

            # Added by the author -- save model + sample
            if self.checkpoint_freq is not None:
                if (i_epoch + 1) % self.checkpoint_freq == 0:
                    save_checkpoint(
                        {
                            "epoch": i_epoch + 1,
                            "arch": self.args.arch,
                            "best_metric": 0,
                            "exp_logger": exp_logger,
                        },
                        self._generator.state_dict(),
                        optimizerG.state_dict(),
                        self.args.dir_logs,
                        i_epoch=i_epoch + 1,
                        save_model=True,
                        save_all_from=None,
                        is_best=True,
                        suffix="generator",
                    )

                    save_checkpoint(
                        {
                            "epoch": i_epoch + 1,
                            "arch": self.args.arch,
                            "best_metric": 0,
                            "exp_logger": exp_logger,
                        },
                        discriminator.state_dict(),
                        optimizerD.state_dict(),
                        self.args.dir_logs,
                        i_epoch=i_epoch + 1,
                        save_model=True,
                        save_all_from=None,
                        is_best=True,
                        suffix="discriminator",
                    )

                    n_samples = int(len(train_data) * 1.05)
                    synthetic_data = self.sample(n_samples)
                    synthetic_data.to_csv(
                        os.path.join(self.args.dir_logs, f"fake_{i_epoch+1:05}.csv"),
                        sep="\t",
                        encoding="utf-8",
                    )

            exp_logger.log_meters("train", n=i_epoch)
            exp_logger.to_json(
                os.path.join(self.args.dir_logs, "logger.json")
            )  # overwrite logger.json after each epoch
            # Added by the author -- save model + sample
