import torch
import numpy as np
import pandas as pd
import zero
import os
from models.tab_ddpm.tab_ddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from models.tab_ddpm.tab_ddpm.utils import FoundNANsError
from models.tab_ddpm.scripts.utils_train import get_model, make_dataset
import models.tab_ddpm.lib as lib
from models.tab_ddpm.lib import round_columns
from engine.datasets import get_dataset


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1] : indices[i]], axis=1)
        t = X[:, indices[i - 1] : indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
    parent_dir,
    real_data_path="data/higgs-small",
    batch_size=2000,
    num_samples=0,
    model_type="mlp",
    model_params=None,
    model_path=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    disbalance=None,
    device=torch.device("cuda:1"),
    seed=0,
    change_val=False,
    steps=10000,
    dataset="adult",
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params["num_classes"],
        is_y_cond=model_params["is_y_cond"],
        change_val=change_val,
    )

    K = np.array(D.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features_ = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params["d_in"] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes("train"),
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(
        torch.from_numpy(D.y["train"]), return_counts=True
    )
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == "fix":
        empirical_class_dist[0], empirical_class_dist[1] = (
            empirical_class_dist[1],
            empirical_class_dist[0],
        )
        x_gen, y_gen = diffusion.sample_all(
            num_samples, batch_size, empirical_class_dist.float(), ddim=False
        )

    elif disbalance == "fill":
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(
                num_samples, batch_size, distrib.float(), ddim=False
            )
            x_gen.append(x_temp)
            y_gen.append(y_temp)

        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(
            num_samples, batch_size, empirical_class_dist.float(), ddim=False
        )

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    num_numerical_features = num_numerical_features + int(
        D.is_regression and not model_params["is_y_cond"]
    )

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        if T_dict["cat_encoding"] == "one-hot":
            X_gen[:, num_numerical_features:] = to_good_ohe(
                D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:]
            )
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])
    else:
        X_cat = None

    if num_numerical_features_ != 0:
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(
            os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True
        )
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params["num_classes"] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)
    else:
        X_num = None

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        # np.save(os.path.join(parent_dir, "X_num_train"), X_num)
    if num_numerical_features < X_gen.shape[1]:
        print("Cat shape: ", X_cat.shape)
        # np.save(os.path.join(parent_dir, "X_cat_train"), X_cat)
    # np.save(os.path.join(parent_dir, "y_train"), y_gen)

    ## added by Author
    G = get_dataset(dataset)
    num_columns = [col for col in G.continuous_columns if col not in G.target]
    cat_columns = [col for col in G.discrete_columns if col not in G.target]
    target_column = G.target
    columns = num_columns + cat_columns + [target_column]

    def concatenate_arrays(x_num, x_cat, y_):
        arrays_to_concat = []

        # Append only the arrays that are not None
        if x_num is not None:
            arrays_to_concat.append(x_num)

        if x_cat is not None:
            arrays_to_concat.append(x_cat)

        # Append the target variable (y_real) after expanding its dimension
        if y_ is not None:
            arrays_to_concat.append(np.expand_dims(y_, axis=1))

        # Concatenate the available arrays along axis 1
        if len(arrays_to_concat) > 0:
            return np.concatenate(arrays_to_concat, axis=1)
        else:
            return None

    X_num_real = (
        np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        if num_numerical_features != 0
        else None
    )
    X_cat_real = (
        np.load(os.path.join(real_data_path, "X_cat_train.npy"), allow_pickle=True)
        if num_numerical_features < X_gen.shape[1]
        else None
    )
    y_real = np.load(os.path.join(real_data_path, "y_train.npy"), allow_pickle=True)
    preprocessed_data = concatenate_arrays(X_num_real, X_cat_real, y_real)
    df_preprocessed = pd.DataFrame(data=preprocessed_data, columns=columns)
    df_preprocessed.to_csv(
        os.path.join(parent_dir, "preprocessed.csv"), sep="\t", encoding="utf-8"
    )

    fake = concatenate_arrays(X_num, X_cat, y_gen)
    df = pd.DataFrame(data=fake, columns=columns)
    df.to_csv(
        os.path.join(parent_dir, f"fake_{steps:05}.csv"), sep="\t", encoding="utf-8"
    )
    ## added by Author
