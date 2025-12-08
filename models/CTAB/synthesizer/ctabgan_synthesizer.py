import os
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    Sigmoid,
    init,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
    LayerNorm,
)
from models.CTAB.synthesizer.transformer import ImageTransformer, DataTransformer
from models.CTAB.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from tqdm import tqdm


from engine.utils.model_utils import save_checkpoint
import engine.utils.train_utils as train_utils
import engine.utils.print_utils as print_utils
import engine.utils.model_utils as model_utils
import engine.utils.eval_utils as eval_utils
import engine.utils.path_utils as path_utils

from engine.custom_loss import (
    CorrectedCorrelationLoss,
    DistributionLoss,
)


class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed):
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        if (st_ed[1] - st_ed[0]) == 1:
            seq += [Linear(dim, 1)]

        elif (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(dim, 1), Sigmoid()]
        else:
            seq += [Linear(dim, (st_ed[1] - st_ed[0]))]

        self.seq = Sequential(*seq)

    def forward(self, input):
        label = None

        if (self.str_end[1] - self.str_end[0]) == 1:
            label = input[:, self.str_end[0] : self.str_end[1]]
        else:
            label = torch.argmax(input[:, self.str_end[0] : self.str_end[1]], axis=-1)

        new_imp = torch.cat(
            (input[:, : self.str_end[0]], input[:, self.str_end[1] :]), 1
        )

        if ((self.str_end[1] - self.str_end[0]) == 2) | (
            (self.str_end[1] - self.str_end[0]) == 1
        ):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == "tanh":
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == "softmax":
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)


def get_st_ed(target_col_index, output_info):
    st = 0
    c = 0
    tc = 0

    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
            if item[2] == "yes_g":
                c += 1
        elif item[1] == "softmax":
            st += item[0]
            c += 1
        tc += 1

    ed = st + output_info[tc][0]

    return (st, ed)


def random_choice_prob_index_sampling(probs, col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list).reshape(col_idx.shape)


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval


class Cond(object):
    def __init__(self, data, output_info):
        self.model = []
        st = 0
        counter = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))
        self.p_sampling = []
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp_sampling = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, : item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype="float32")
        mask = np.zeros((batch, self.n_col), dtype="float32")
        mask[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype="float32")
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)

        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == "tanh":
            st += item[0]
            continue

        elif item[1] == "softmax":
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none"
            )
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers) - 2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)


class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)


def determine_layers_disc(side, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []
    num_c = num_channels
    num_s = side / 2
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c * 2
        num_s = num_s / 2

    layers_D = []

    for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            LayerNorm(ln),
            LeakyReLU(0.2, inplace=True),
        ]

    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)]

    return layers_D


def determine_layers_gen(side, random_dim, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []

    num_c = num_channels * (2 ** (len(layer_dims) - 2))
    num_s = int(side / (2 ** (len(layer_dims) - 1)))
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c / 2
        num_s = num_s * 2

    layers_G = [
        ConvTranspose2d(
            random_dim,
            layer_dims[-1][0],
            layer_dims[-1][1],
            1,
            0,
            output_padding=0,
            bias=False,
        )
    ]

    for prev, curr, ln in zip(
        reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms
    ):
        layers_G += [
            LayerNorm(ln),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True),
        ]
    return layers_G


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (
        torch.sin(val * omega) / so
    ) * high

    return res


def calc_gradient_penalty_slerp(
    netD, real_data, fake_data, transformer, device="cpu", lambda_=10
):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1, device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_

    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class CTABGANSynthesizer:
    def __init__(
        self,
        args,
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=500,
        epochs=150,
        is_loss_corr=0,
        is_loss_dwp=0,
        n_moment_loss_dwp=4,
        checkpoint_freq=50,
    ):
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Added by the author
        self.args = args
        self.checkpoint_freq = checkpoint_freq
        self.is_loss_corr = is_loss_corr
        self.is_loss_dwp = is_loss_dwp
        self.n_moment_loss_dwp = n_moment_loss_dwp
        self._device = torch.device("cuda")
        # Added by the author

    def fit(
        self,
        train_data=pd.DataFrame,
        exp_logger=None,
        data_prep=None,
        discrete_columns=(),
        categorical=[],
        mixed={},
        general=[],
        non_categorical=[],
        type={},
    ):
        # Added by the author -- corr + dwp
        self._corrcoef_original = torch.corrcoef(
            torch.transpose(
                torch.tensor(train_data.to_numpy(), dtype=torch.float32), 0, 1
            ).to(self._device)
        )
        self._corrcoef_original = torch.nan_to_num(self._corrcoef_original, nan=0)
        self.df = train_data.copy()
        # Added by the author -- corr + dwp

        problem_type = None
        target_index = None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self._transformer = DataTransformer(
            train_data=train_data,
            categorical_list=categorical,
            mixed_dict=mixed,
            general_list=general,
            non_categorical_list=non_categorical,
        )
        self._transformer.fit()
        train_data = self._transformer.transform(train_data.values)
        data_sampler = Sampler(train_data, self._transformer.output_info)
        data_dim = self._transformer.output_dim
        self.cond_generator = Cond(train_data, self._transformer.output_info)

        sides = [4, 8, 16, 24, 32, 64]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        sides = [4, 8, 16, 24, 32, 64]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        layers_G = determine_layers_gen(
            self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels
        )
        layers_D = determine_layers_disc(self.dside, self.num_channels)

        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)
        optimizer_params = dict(
            lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale
        )
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        # Added by the author -- load
        if self.args.resume:
            try:
                (
                    self.args.start_epoch,
                    best_metric,
                    exp_logger,
                    is_loadable_resume,
                ) = model_utils.load_checkpoint(
                    self.generator, optimizerG, self.args.dir_logs, suffix="generator"
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

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index != None:
            st_ed = get_st_ed(target_index, self._transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        epsilon = 0
        epoch = 0
        steps = 0
        ci = 1

        # Added by the author -- logger
        meters = exp_logger.reset_meters("train")
        start_epoch = time.time()
        # Added by the author -- logger

        train_data_tensor = torch.tensor(train_data).to(self._device)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i_epoch in range(self.epochs):
            # rerun i_epoch if an error appears
            # is_run = 0

            # while not is_run:
            #     try:
            if self.args.resume and i_epoch < self.args.start_epoch:
                continue
            for id_ in range(steps_per_epoch):
                for _ in range(ci):
                    noisez = torch.randn(
                        self.batch_size, self.random_dim, device=self.device
                    )
                    condvec = self.cond_generator.sample_train(self.batch_size)

                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(self.device)
                    m = torch.from_numpy(m).to(self.device)
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez = noisez.view(
                        self.batch_size,
                        self.random_dim + self.cond_generator.n_opt,
                        1,
                        1,
                    )

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]

                    real = torch.from_numpy(real.astype("float32")).to(self.device)

                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self._transformer.output_info)

                    fake_cat = torch.cat([fakeact, c], dim=1)
                    real_cat = torch.cat([real, c_perm], dim=1)

                    real_cat_d = self.Dtransformer.transform(real_cat)
                    fake_cat_d = self.Dtransformer.transform(fake_cat)

                    optimizerD.zero_grad()

                    d_real, _ = discriminator(real_cat_d)
                    d_real = -torch.mean(d_real)
                    d_real.backward()

                    d_fake, _ = discriminator(fake_cat_d)
                    d_fake = torch.mean(d_fake)
                    d_fake.backward()

                    # Added by the author -- logger
                    loss_d = -(d_real - d_fake)
                    meters["loss_d"].update(loss_d.item(), fake.size(0))
                    # Added by the author -- logger

                    pen = calc_gradient_penalty_slerp(
                        discriminator,
                        real_cat,
                        fake_cat,
                        self.Dtransformer,
                        self.device,
                    )

                    pen.backward()

                    optimizerD.step()

                noisez = torch.randn(
                    self.batch_size, self.random_dim, device=self.device
                )

                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size,
                    self.random_dim + self.cond_generator.n_opt,
                    1,
                    1,
                )

                optimizerG.zero_grad()

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self._transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.Dtransformer.transform(fake_cat)

                y_fake, info_fake = discriminator(fake_cat)

                cross_entropy = cond_loss(faket, self._transformer.output_info, c, m)

                _, info_real = discriminator(real_cat_d)

                # Added by the author -- corr + dwp
                df_fake_inverse, _ = self._transformer.inverse_transform(
                    fakeact.detach().cpu().numpy()
                )
                df_fake_inverse = data_prep.inverse_prep(df_fake_inverse)

                if (
                    self.args.loss_version > 0
                    and self.is_loss_corr
                    # Added by the author -- only dis. loss
                    and self.args.loss_version != 5
                    # Added by the author -- only dis. loss
                ):
                    loss_corr = self.is_loss_corr * CorrectedCorrelationLoss()(
                        train_data_tensor, fakeact
                    )  # version 2 and 3
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
                    )  # version 2
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
                g = -torch.mean(y_fake) + cross_entropy + loss_corr + loss_dwp
                # Added by the author -- corr + dwp

                # Added by the author -- logger
                meters["loss_g"].update(g.item(), fake.size(0))
                # Added by the author -- logger

                g.backward(retain_graph=True)
                loss_mean = torch.norm(
                    torch.mean(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.mean(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_std = torch.norm(
                    torch.std(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.std(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                if problem_type:
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self._transformer.output_info)

                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fakeact)

                    c_loss = CrossEntropyLoss()

                    if (st_ed[1] - st_ed[0]) == 1:
                        c_loss = SmoothL1Loss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                        real_label = torch.reshape(real_label, real_pre.size())
                        fake_label = torch.reshape(fake_label, fake_pre.size())

                    elif (st_ed[1] - st_ed[0]) == 2:
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)

                    loss_cc = c_loss(real_pre, real_label)
                    loss_cg = c_loss(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()

            # measure elapsed time
            meters["epoch_time"].update(time.time() - start_epoch)
            start_epoch = time.time()

            curr_epoch = i_epoch + 1
            txt_epoch = "{:<10s}".format("Epoch:") + "{curr_epoch:04}".format(
                curr_epoch=curr_epoch
            )
            txt_epoch_time = "{:<8s}".format("Etime:") + "{epoch_time.val:.4f}".format(
                epoch_time=meters["epoch_time"]
            )
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
                ) + "{metric_corr.avg:.4f}".format(metric_corr=meters["metric_corr"])
            else:
                txt_metric_corr = "{:<8s}".format(
                    "MCorr:"
                ) + "{metric_corr.avg:.4f}".format(metric_corr=meters["metric_corr"])

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
            print(f"{txt_epoch:<25s} {txt_epoch_time:<25s} {txt_d:<25s} {txt_g:<25s}")
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
                    self.generator.state_dict(),
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
                n_samples = int(len(train_data) * 1.2)
                synthetic_data = self.sample(n_samples)
                synthetic_data = data_prep.inverse_prep(synthetic_data)
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

            # is_run = 1

            # except Exception as e:
            #     print(f"Error in epoch {i_epoch}: {e}")
            #     print("Rerunning epoch...")

    def sample(self, n):
        self.generator.eval()

        output_info = self._transformer.output_info
        steps = n // self.batch_size + 1

        data = []

        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(
                self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1
            )

            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result, resample = self._transformer.inverse_transform(data)

        while len(result) < n:
            data_resample = []
            steps_left = resample // self.batch_size + 1

            for i in range(steps_left):
                noisez = torch.randn(
                    self.batch_size, self.random_dim, device=self.device
                )
                condvec = self.cond_generator.sample(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1
                )

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res, resample = self._transformer.inverse_transform(data_resample)
            result = np.concatenate([result, res], axis=0)

        return result[0:n]
