"""TVAE module."""

import os
import numpy as np
import pandas as pd
import time
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from engine.ctgan_data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from engine.utils.model_utils import save_checkpoint
import engine.utils.train_utils as train_utils
import engine.utils.model_utils as model_utils
import engine.utils.eval_utils as eval_utils
import engine.utils.print_utils as print_utils

from engine.custom_loss import (
    CorrectedCorrelationLoss,
    DistributionLoss,
)


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""

        # # for testing
        # i = 0
        # for name, layer in self.seq.named_children():
        #     if i < 1:
        #         x = layer(input_)
        #     else:
        #         x = layer(x)
        #     i += 1
        #     # print(f"Layer Output: {x.shape}")

        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != "softmax":
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed],
                        torch.argmax(x[:, st:ed], dim=-1),
                        reduction="sum",
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        args,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=True,
        is_loss_corr=0,
        is_loss_dwp=0,
        n_moment_loss_dwp=4,
        checkpoint_freq=50,
    ):
        # Added by the author
        self.args = args
        self.checkpoint_freq = checkpoint_freq
        self.is_loss_corr = is_loss_corr
        self.is_loss_dwp = is_loss_dwp
        self.n_moment_loss_dwp = n_moment_loss_dwp
        # Added by the author

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=["Epoch", "Batch", "Loss"])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

    @random_state
    def fit(self, train_data, exp_logger, discrete_columns=(), id_columns=None):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        # Added by the author -- corr + dwp
        self._corrcoef_original = torch.corrcoef(
            torch.transpose(
                torch.tensor(train_data.to_numpy(), dtype=torch.float32), 0, 1
            ).to(self._device)
        )
        self._corrcoef_original = torch.nan_to_num(self._corrcoef_original, nan=0)
        self.df = train_data.copy()
        # Added by the author -- corr + dwp

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(
            torch.from_numpy(train_data.astype("float32")).to(self._device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(
            self._device
        )
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(
            self._device
        )
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale,
        )

        # Added by the author -- logger
        exp_logger.info["encoder_model_params"] = train_utils.params_count(encoder)
        exp_logger.info["decoder_model_params"] = train_utils.params_count(self.decoder)
        # Added by the author -- logger

        # Added by the author -- load
        if self.args.resume:
            try:
                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    encoder, optimizerAE, self.args.dir_logs, suffix="encoder"
                )

                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    self.decoder,
                    optimizerAE,
                    self.args.dir_logs,
                    suffix="decoder",
                )
            except:
                print("Something is wrong. Can't load last model")
                self.args.start_epoch = 0
        # Added by the author -- load

        # Added by the author -- logger
        meters = exp_logger.reset_meters("train")
        start_epoch = time.time()
        # Added by the author -- logger

        for i_epoch in range(self.epochs):
            if self.args.resume and i_epoch < self.args.start_epoch:
                continue
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )

                # Added by the author -- logger
                meters["loss_1"].update(loss_1.item(), real.size(0))
                meters["loss_2"].update(loss_2.item(), real.size(0))
                # Added by the author -- logger

                # Added by the author -- corr + dwp
                df_fake_inverse = self.transformer.inverse_transform(
                    rec.detach().cpu().numpy()
                )
                if (
                    self.args.loss_version > 0
                    and self.is_loss_corr
                    # Added by the author -- only dis. loss
                    and self.args.loss_version != 5
                    # Added by the author -- only dis. loss
                ):
                    loss_corr = self.is_loss_corr * CorrectedCorrelationLoss()(
                        real, rec
                    )  # version 2 and 3
                    meters["loss_corr"].update(loss_corr.item(), rec.size(0))
                else:
                    loss_corr = 0
                    meters["loss_corr"].update(0, rec.size(0))

                if (
                    self.args.loss_version > 0
                    and self.is_loss_dwp
                    # Added by the author -- only cor. loss
                    and self.args.loss_version != 4
                    # Added by the author -- only cor. loss
                ):
                    loss_dwp = self.is_loss_dwp * DistributionLoss()(
                        real,
                        rec,
                        alpha=1,
                        n=self.n_moment_loss_dwp,
                    )  # version 2
                    meters["loss_dwp"].update(loss_dwp.item(), rec.size(0))
                else:
                    loss_dwp = 0
                    meters["loss_dwp"].update(0, rec.size(0))

                metric_corr = eval_utils.compute_diff_correlation(
                    self.df, df_fake_inverse
                )
                meters["metric_corr"].update(metric_corr, 1)
                metric_dwp, _, _ = eval_utils.compute_dwp(
                    self.df, df_fake_inverse, discrete_columns=discrete_columns
                )
                meters["metric_dwp"].update(metric_dwp, 1)

                # total loss
                loss = loss_1 + loss_2 + loss_corr + loss_dwp
                # Added by the author -- corr + dwp

                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            # measure elapsed time
            meters["epoch_time"].update(time.time() - start_epoch)
            start_epoch = time.time()

            if self.verbose:
                curr_epoch = i_epoch + 1
                txt_epoch = "{:<10s}".format("Epoch:") + "{curr_epoch:04}".format(
                    curr_epoch=curr_epoch
                )
                txt_epoch_time = "{:<8s}".format(
                    "Etime:"
                ) + "{epoch_time.val:.4f}".format(epoch_time=meters["epoch_time"])
                txt_g = "{:<8s}".format("Loss_1:") + "{loss_1.avg:.4f}".format(
                    loss_1=meters["loss_1"]
                )
                txt_d = "{:<8s}".format("Loss_2:") + "{loss_2.avg:.4f}".format(
                    loss_2=meters["loss_2"]
                )
                txt_corr = "{:<8s}".format("LossC:") + "{loss_corr.avg:.4f}".format(
                    loss_corr=meters["loss_corr"]
                )
                txt_dwp = "{:<8s}".format("LossP:") + "{loss_dwp.avg:.4f}".format(
                    loss_dwp=meters["loss_dwp"]
                )

                if meters["loss_1"].avg < 0:
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

                if meters["loss_2"].avg < 0:
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

            # Added by the author -- save model + sample
            # if (
            #     self.checkpoint_freq is not None
            #     and (i_epoch + 1) % self.checkpoint_freq == 0

            # ):
            if i_epoch + 1 == self.epochs:  # save storage
                print_utils.print_processing("save models")
                save_checkpoint(
                    {
                        "epoch": i_epoch + 1,
                        "arch": self.args.arch,
                        "best_metric": 0,
                        "exp_logger": exp_logger,
                    },
                    encoder.state_dict(),
                    optimizerAE.state_dict(),
                    self.args.dir_logs,
                    i_epoch=i_epoch + 1,
                    save_model=True,
                    save_all_from=None,
                    is_best=True,
                    suffix="encoder",
                )

                save_checkpoint(
                    {
                        "epoch": i_epoch + 1,
                        "arch": self.args.arch,
                        "best_metric": 0,
                        "exp_logger": exp_logger,
                    },
                    self.decoder.state_dict(),
                    optimizerAE.state_dict(),
                    self.args.dir_logs,
                    i_epoch=i_epoch + 1,
                    save_model=True,
                    save_all_from=None,
                    is_best=True,
                    suffix="decoder",
                )

                print_utils.print_processing("generate synthetic data")
                n_samples = int(len(train_data) * 1.2)
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

        if self.args.row_number_full is not None:
            print_utils.print_processing("generate full synthetic data")
            n_samples = int(self.args.row_number_full * 1.05)
            synthetic_data = self.sample(n_samples)
            synthetic_data.to_csv(
                os.path.join(self.args.dir_logs, f"synthetic_full.csv"),
                sep="\t",
                encoding="utf-8",
            )

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
