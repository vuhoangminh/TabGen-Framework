import os
import numpy as np
import torch
import shutil
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Autoencoder(torch.nn.Module):
    def __init__(self, dim_input, dim_layer_1, dim_layer_2):
        super().__init__()

        self.dim_input = dim_input
        self.dim_layer_1 = dim_layer_1
        self.dim_layer_2 = dim_layer_2

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(dim_input, dim_layer_1),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_layer_1, dim_layer_2),
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(dim_layer_2, dim_layer_1),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_layer_1, dim_input),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def perform_ae(
    X,
    autoencoder_input=4,
    autoencoder_output=3,
    list_encode=[
        "subproj",
        "sex",
        "llkk_txt",
        "llkk_county_letter",
        "vdc",
        "bmi",
        "predict_cohort",
    ],
    epochs=1000,
    min_delta=0.001,
    patience=50,
):
    feat_confidential = list_encode[:autoencoder_input]
    feat_groups = [feat_confidential]
    list_encoded = []

    # dim
    g = feat_groups[0]
    dim_input = len(g)
    dim_layer_1 = round((autoencoder_input + autoencoder_output) / 2)
    dim_layer_2 = autoencoder_output

    X_tmp = X.loc[:, g]
    # model
    model = Autoencoder(dim_input, dim_layer_1, dim_layer_2)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # dataloader
    X_train, X_test = train_test_split(X_tmp, test_size=0.3, random_state=2023)
    X_train = torch.Tensor(X_train.to_numpy())
    X_test = torch.Tensor(X_test.to_numpy())
    X_full = torch.Tensor(X_tmp.to_numpy())

    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1024)
    test_dataset = TensorDataset(X_test, X_test)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1024)
    full_dataset = TensorDataset(X_full, X_full)
    full_loader = DataLoader(full_dataset, shuffle=False, batch_size=1024)

    path_last_best_ae_model = f"database/autoencoder/last_best_ae_model_in-{dim_input}_l1-{dim_layer_1}_l2-{dim_layer_2}.pt"

    if not os.path.exists(path_last_best_ae_model):
        print(f"cant find {path_last_best_ae_model}")
        print("start training")

        # init
        best_loss = np.inf
        train_losses = []
        test_losses = []
        count_patience = 0
        for epoch in range(epochs):
            train_loss = 0
            for x, _ in train_loader:
                # Output of Autoencoder
                reconstructed, encoded = model(x)

                # Calculating the loss function
                loss = loss_function(reconstructed, x)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                train_losses.append(loss)
                train_loss += loss.detach().cpu().numpy()

            train_loss /= len(train_loader)

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    # Output of Autoencoder
                    reconstructed, encoded = model(x)

                    # Calculating the loss function
                    loss = loss_function(reconstructed, x)

                    # Storing the losses in a list for plotting
                    test_losses.append(loss)
                    test_loss += loss.detach().cpu().numpy()

            test_loss /= len(test_loader)

            print("-" * 100)
            print(
                f"epoch: {epoch} \t train_loss: {train_loss:.4f} \t test_loss: {test_loss:.4f} \t best_loss: {best_loss:.4f} \t patience: {count_patience}"
            )

            if test_loss >= best_loss or (best_loss - test_loss) < min_delta:
                count_patience += 1

            if test_loss < best_loss:
                if (best_loss - test_loss) >= min_delta:
                    count_patience = 0  # reset count_patience
                best_loss = test_loss
                path_best_ae_model = f"database/autoencoder/best_ae_model_in-{dim_input}_l1-{dim_layer_1}_l2-{dim_layer_2}.pt"

                torch.save(model, path_best_ae_model)
                print("update best model")

            if count_patience >= patience:
                break

        # copy the last best model
        shutil.copyfile(path_best_ae_model, path_last_best_ae_model)

    else:
        print("skip training")

    # Load
    model = torch.load(path_last_best_ae_model)
    model.eval()

    with torch.no_grad():
        for x, _ in full_loader:
            # Output of Autoencoder
            reconstructed, encoded = model(x)
            list_encoded.append(encoded.detach().cpu().numpy())

    # f_confidential_encoded = np.hstack(list_encoded)
    f_confidential_encoded = np.vstack(list_encoded)

    feat_not_confidential = [i for i in list(X.columns) if i not in feat_confidential]
    X_encoded = X.loc[:, feat_not_confidential]

    for i in range(autoencoder_output):
        X_encoded[f"Confidential{i}"] = f_confidential_encoded[:, i]

    return X_encoded
