import torch
import time


class CorrectedCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(CorrectedCorrelationLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_loss(self, X, Y):
        """
        Calculates the correlation loss between X and Y.

        Args:
            X (torch.Tensor): The first input tensor of shape (batch_size, num_features).
            Y (torch.Tensor): The second input tensor of the same shape as X.

        Returns:
            torch.Tensor: The calculated correlation loss.
        """
        X_normalized = (X - X.mean(dim=0)) / (X.std(dim=0) + self.epsilon)
        Y_normalized = (Y - Y.mean(dim=0)) / (Y.std(dim=0) + self.epsilon)

        # Compute the cross-correlation matrix
        X_cross_corr_matrix = torch.mm(X_normalized.T, X_normalized) / X.shape[0]
        Y_cross_corr_matrix = torch.mm(Y_normalized.T, Y_normalized) / Y.shape[0]

        # Exclude diagonal elements (self-correlations)
        X_cross_corr_matrix -= torch.diag(X_cross_corr_matrix.diag())
        Y_cross_corr_matrix -= torch.diag(Y_cross_corr_matrix.diag())

        # Calculate the squared difference between cross-correlations
        diff_matrix = X_cross_corr_matrix - Y_cross_corr_matrix
        loss = (diff_matrix**2).sum() / (X.shape[1] * (X.shape[1] - 1))

        return loss

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = self.compute_loss(X, Y)
        return loss


class DistributionLoss(torch.nn.Module):
    """
    This class implements a loss function that compares the distribution moments
    (mean, variance, higher-order moments) between two tensors.

    Args:
        None: This class does not require any arguments during initialization.

    Returns:
        torch.Tensor: The total loss calculated based on the differences in distribution moments.
    """

    def __init__(self):
        super(DistributionLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_moments_along_rows(self, data, rank, dim=0):
        """
        Calculates the specified moment (mean, variance, higher-order moments)
        along the specified dimension (default: 0) of a 2D tensor.

        Args:
            data (torch.Tensor): The input tensor (must be 2D).
            rank (int): The order of the moment to calculate (e.g., 1 for mean, 2 for variance).
            dim (int, optional): The dimension along which to reduce (default: 0 for rows).

        Raises:
            ValueError: If the input data is not a 2D tensor.

        Returns:
            torch.Tensor: The calculated moment along the specified dimension.
        """

        if len(data.shape) > 2:
            raise ValueError("Input data must be a 1D or 2D tensors.")

        # Reduce moment along rows
        if rank == 1:
            return torch.mean(data, dim=dim)
        elif rank == 2:
            return torch.var(data, dim=dim)
        else:
            centered_data = data - torch.mean(
                data, dim=dim
            )  # Centering for higher moments
            return torch.mean(
                (centered_data / (torch.std(data, dim=dim) + self.epsilon)) ** rank,
                dim=dim,
            )  # add small value to avoid division by 0

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float = 1.0,
        n: int = 4,
    ) -> torch.Tensor:
        """
        Calculates the distribution loss between two tensors X and Y.

        Args:
            X (torch.Tensor): The real tensor.
            Y (torch.Tensor): The synthetic tensor.
            alha/lambda (float, optional): Weighting factor for the loss terms (default: 1.0).
            n (int, optional): The number of moments to compare (default: 4).

        Raises:
            AssertionError: If any element in X or Y is NaN.

        Returns:
            torch.Tensor: The total loss calculated based on the moment differences.
        """

        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = 0
        for rank in range(1, n + 1):  # Start from 1 to exclude mean (rank 0)
            moment_real = self.compute_moments_along_rows(X, rank)
            moment_syn = self.compute_moments_along_rows(Y, rank)
            l = (
                1 - (moment_syn + self.epsilon) / (moment_real + self.epsilon)
            ) ** 2  # add small value to avoid division by 0
            loss += alpha / rank * l

        return loss.mean()


def test_loss_function(loss_fn, input1, input2):
    """
    Tests the custom loss function and checks requires_grad for the first input.

    Args:
        loss_fn: Your custom loss function (callable).
        input1 (torch.Tensor): The first input to the loss function.
        input2 (torch.Tensor): The second input to the loss function.
    """

    start = time.time()

    # Set requires_grad to True for input1 (if not already set)
    if not input1.requires_grad:
        input1.requires_grad = True

    # Calculate the loss
    output = loss_fn(input1, input2)

    # Print the loss value
    print(f"Loss: {output.item()}")

    # Check requires_grad for input1
    print(f"Input 1 requires_grad: {input1.requires_grad}")
    print(f"Output requires_grad: {output.requires_grad}")

    end = time.time()
    print(f"Elapsed time is:", end - start)


def main():
    # Example usage (assuming your loss function is named 'my_loss_function')
    ncols = 100
    data1 = (
        torch.randn(40000, ncols, requires_grad=True) * 1
    )  # Ensure requires_grad for input1
    data2 = torch.randn(25000, ncols) * 2

    test_loss_function(DistributionLoss(), data1, data2)
    test_loss_function(CorrectedCorrelationLoss(), data1, data2)


if __name__ == "__main__":
    main()
