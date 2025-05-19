import torch

class LinearModel:
    def __init__(self):
        self.w = None

    def score(self, X):
        """ Compute the score for each data point """
        if self.w is None:
            self.w = torch.rand((X.size()[1]))

        return X @ self.w

    def predict(self, X):
        """ Compute predictions (0 or 1) """
        return (self.score(X) > 0).float()

class Perceptron(LinearModel):
    def loss(self, X, y):
        """ Compute the misclassification rate """
        y_ = 2 * y - 1  # Convert y to {-1, 1}
        s = self.score(X)
        return (s * y_ <= 0).float().mean()

    def grad(self, X, y):
        """
        Compute the perceptron gradient update for a **single row** X.
        Formula: -1 * [s_i (2y_i - 1) < 0] * y_i * X_i
        """
        y_ = 2 * y - 1  # Convert y to {-1, 1}
        s = self.score(X)
        incorrect_mask = (s * y_ < 0).float().unsqueeze(1)  # Mask for misclassified points
        grad = -incorrect_mask * y_.unsqueeze(1) * X
        return grad.mean(dim=0)  # Return the mean gradient update

class PerceptronOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y):
        """
        Perform one perceptron update step:
        1. Compute loss (not used directly for update, but called for consistency).
        2. Compute gradient and update model weights.
        """
        _ = self.model.loss(X, y)  # Call loss function (though we don't use its value)
        grad_update = self.model.grad(X, y)  # Compute gradient
        self.model.w -= grad_update  # Update weights
