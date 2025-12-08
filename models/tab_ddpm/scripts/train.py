from copy import deepcopy
import torch
import os
import numpy as np
from models.tab_ddpm.tab_ddpm import GaussianMultinomialDiffusion
from models.tab_ddpm.scripts.utils_train import get_model, make_dataset, update_ema
import models.tab_ddpm.lib as lib
import pandas as pd
from models.tab_ddpm.tab_ddpm.utils import index_to_log_onehot


class Trainer:
    def __init__(
        self,
        diffusion,
        train_iter,
        lr,
        weight_decay,
        steps,
        device=torch.device("cuda:1"),
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(
            columns=[
                "step",
                "mloss",
                "gloss",
                "dloss",
                "closs",
                "dloss_num",
                "closs_num",
                "loss",
            ]
        )
        self.log_every = 100
        self.print_every = 100
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        ### Added by the author
        loss_multi, loss_gauss, loss_dwp, loss_corr, loss_dwp_num, loss_corr_num = (
            self.diffusion.mixed_loss(x, out_dict)
        )
        ### Added by the author
        loss = (
            loss_multi
            + loss_gauss
            + loss_dwp
            + loss_corr
            + loss_dwp_num
            + loss_corr_num
        )
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss, loss_dwp, loss_corr, loss_dwp_num, loss_corr_num

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        ## Added
        curr_loss_dwp = 0.0
        curr_loss_corr = 0.0
        curr_loss_dwp_num = 0.0
        curr_loss_corr_num = 0.0
        ## Added

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {"y": out_dict}

            ## Added
            (
                batch_loss_multi,
                batch_loss_gauss,
                batch_loss_dwp,
                batch_loss_corr,
                batch_loss_dwp_num,
                batch_loss_corr_num,
            ) = self._run_step(x, out_dict)
            ## Added

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if batch_loss_dwp > 0:
                curr_loss_dwp += batch_loss_dwp.item() * len(x)

            if batch_loss_corr > 0:
                curr_loss_corr += batch_loss_corr.item() * len(x)

            if batch_loss_dwp_num > 0:
                curr_loss_dwp_num += batch_loss_dwp_num.item() * len(x)

            if batch_loss_corr_num > 0:
                curr_loss_corr_num += batch_loss_corr_num.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)

                ## Added
                dloss = np.around(curr_loss_dwp / curr_count, 4)
                closs = np.around(curr_loss_corr / curr_count, 4)
                dloss_num = np.around(curr_loss_dwp_num / curr_count, 4)
                closs_num = np.around(curr_loss_corr_num / curr_count, 4)
                ## Added

                if (step + 1) % self.print_every == 0:
                    sumloss = mloss + gloss + dloss + closs + dloss_num + closs_num
                    formatted_string = (
                        f"Step {(step + 1)}/{self.steps}    "
                        f"MLoss : {mloss:10.4f}     "
                        f"GLoss : {gloss:10.4f}     "
                        f"DLoss : {dloss:10.4f}     "
                        f"CLoss : {closs:10.4f}     "
                        f"DLossN: {dloss_num:10.4f}     "
                        f"CLossN: {closs_num:10.4f}     "
                        f"Sum: {sumloss:10.4f}"
                    )
                    print(formatted_string)
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    dloss,
                    closs,
                    dloss_num,
                    closs_num,
                    mloss + gloss + dloss + closs,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

                ## Added
                curr_loss_dwp = 0.0
                curr_loss_corr = 0.0
                curr_loss_dwp_num = 0.0
                curr_loss_corr_num = 0.0
                ## Added

            update_ema(
                self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()
            )

            step += 1


def train(
    parent_dir,
    real_data_path="data/higgs-small",
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    model_type="mlp",
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    device=torch.device("cuda:0"),
    seed=0,
    change_val=False,
    ### Added by the author
    loss_version=0,
    is_loss_corr=0,
    is_loss_dwp=0,
    n_moment_loss_dwp=0,
    is_loss_num=0,
    is_loss_corr_num=0,
    is_loss_dwp_num=0,
    n_moment_loss_dwp_num=0,
    num_samples=0,
    ### Added by the author
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params["num_classes"],
        is_y_cond=model_params["is_y_cond"],
        change_val=change_val,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])
    print(K)

    num_numerical_features = (
        dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    )
    d_in = np.sum(K) + num_numerical_features
    model_params["d_in"] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes("train"),
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(
        dataset, split="train", batch_size=batch_size
    )

    ## Added
    _, empirical_class_dist = torch.unique(
        torch.from_numpy(dataset.y["train"]), return_counts=True
    )
    if dataset.X_num is None:
        train_data_tensor = np.concatenate(
            (
                index_to_log_onehot(torch.from_numpy(dataset.X_cat["train"]), K),
                np.expand_dims(dataset.y["train"], axis=1),
            ),
            axis=1,
        )
    elif dataset.X_cat is None:
        train_data_tensor = np.concatenate(
            (
                dataset.X_num["train"],
                np.expand_dims(dataset.y["train"], axis=1),
            ),
            axis=1,
        )
    else:
        train_data_tensor = np.concatenate(
            (
                dataset.X_num["train"],
                index_to_log_onehot(torch.from_numpy(dataset.X_cat["train"]), K),
                np.expand_dims(dataset.y["train"], axis=1),
            ),
            axis=1,
        )
    ## Added

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        ### Added by the author
        is_loss_corr=is_loss_corr,
        is_loss_dwp=is_loss_dwp,
        n_moment_loss_dwp=n_moment_loss_dwp,
        is_loss_num=is_loss_num,
        is_loss_corr_num=is_loss_corr_num,
        is_loss_dwp_num=is_loss_dwp_num,
        n_moment_loss_dwp_num=n_moment_loss_dwp_num,
        empirical_class_dist=empirical_class_dist,
        num_samples=num_samples,
        train_data_tensor=torch.tensor(train_data_tensor).to(device),
        ### Added by the author
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, "loss.csv"), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, "model.pt"))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, "model_ema.pt"))
