import torch
import pandas as pd
import numpy as np
    
class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        s = X @ self.w
        return s
    
class LogisticRegression(LinearModel):
        # loss takes the data frame matrix (X) and the labels (y)
        #return corss entropy loss of logistic model!

    def loss(self,X, y):
        """
        Computes the cross entropy loss of the logistic regression model, taking in a data matrix X and 
        labels y. This function utilizes the sigmoid function in order to make linear regression work for binary
        classification. also adds a small value to logs in order to avoid nan errors. 

        ARGUMENTS:
        X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

        y, torch.tensor: the labels for the data. y.size() == (p),
            where p is the number of data points. 

        RETURNS: 
            entropy_loss: a float denoting the cross entropy loss of the model. 
        """
        s = self.score(X)
        sigma_s = 1/(1+torch.exp(-s)) #avoid nan w small number
        entropy_loss = (-y*torch.log(sigma_s + 1e-8) - (1-y)*torch.log(1-sigma_s + 1e-8)).mean()
        return entropy_loss
                
    def grad(self,X,y):
        """
        Computes the gradient of the the loss function.

        ARGUMENTS:
        X, torch.Tensor: the feature matrix. X_size() == (n,p), 
        where n is the numner of the data points and p is the 
        number of features. This implementation always assumes 
        that the final column of X is  a constant column of 1s. 

        y, torch.tensor: the labels for the Feature matrix X.
        y.size() == (n), where n is the number of data points.

        RETURNS:
        gradient: the gradient vector of the loss function, a 
        torch.tensor of size(p).
        """
        s = self.score(X)
        sigma_s = 1/(1+torch.exp(-s))
        n =X.size(0)
        gradient = (1/n)*X.T @(sigma_s - y)
        return gradient
        

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        gradient = self.model.grad(X, y) # get the gradient from the model
        """
        Takes a step in gradient descent with momentum. The
        first step is done with vanilla gradient descent.
        From there the previous gradient is stored for momentum.

        Arguments:
        X, torch.Tensor: the feature matrix. X_size() == (n,p), 
        where n is the numner of the data points and p is the 
        number of features. This implementation always assumes 
        that the final column of X is  a constant column of 1s. 

        y, torch.tensor: the labels for the Feature matrix X.
        y.size() == (n), where n is the number of data points.

        alpha, float:
        the learning rate of gradient descent.

        beta, float:
        the parameter to determine the strength of momentum in
        the gradient descent. 


        """

        if self.prev_w == None: #if there are no previous weights use "normal" descent
            self.prev_w = self.model.w.clone()
            self.model.w = self.model.w - (alpha * gradient)
        else: #do momentum otherwise
            temp = self.model.w.clone()
            self.model.w = self.model.w - (alpha * gradient) + beta*(self.model.w - self.prev_w)
            self.prev_w = temp
        return(self.model.w) # returning updated weights. 


class KernelLogisticRegression:
    def __init__(self, kernel_func, lam=0.1, **kernel_kwargs):
        """
        Initializes the kernel logistic regression model.

        Args:
            kernel_func (callable): A positive-definite kernel function. Must take two input matrices.
            lam (float): Regularization strength for L1 penalty on dual weights.
            kernel_kwargs: Additional keyword arguments to pass to the kernel function.
        """
        self.kernel_func = kernel_func
        self.kernel_kwargs = kernel_kwargs
        self.lam = lam
        self.a = None  # Dual weight vector
        self.X_train = None  # training data :P

    def score(self, X):
        """
        Computes the score vector s for a given input X using the kernel trick:
            s = K(X, X_train)^T @ a --> m,n

        Args:
            X (torch.Tensor): Input data of shape (m, p)

        Returns:
            s (torch.Tensor): Score vector of shape (m,)
        """
        K = self.kernel_func(X, self.X_train, **self.kernel_kwargs)  # (m, n)
        s = K @ self.a  # (m,)
        return s

    def loss(self, X, y):
        """
        Computes the L1-regularized cross-entropy loss for kernel logistic regression:
            L(a) = -1/m * summation [y log(sigma(s)) + (1 - y) log(1 - sigma(s))] + lambda||a||

        Args:
            X (torch.Tensor): Feature matrix of shape (m, p)
            y (torch.Tensor): Labels of shape (m,)

        Returns:
            loss (float): Scalar loss value
        """
        s = self.score(X)
        sigma_s = torch.sigmoid(s)
        cross_entropy = (-y * torch.log(sigma_s + 1e-8) - (1 - y) * torch.log(1 - sigma_s + 1e-8)).mean()
        reg = self.lam * torch.norm(self.a, p=1)
        return cross_entropy + reg

    def grad(self, X, y):
        """
        Computes the gradient of the L1-regularized loss with respect to dual weights a.

        Args:
            X (torch.Tensor): Input features of shape (m, p)
            y (torch.Tensor): Labels of shape (m,)

        Returns:
            grad (torch.Tensor): Gradient vector of shape (n,), where n is number of training samples.
        """
        K = self.kernel_func(X, self.X_train, **self.kernel_kwargs)  # (m, n)
        s = K @ self.a  # (m,)
        sigma_s = torch.sigmoid(s)
        grad = (1 / X.shape[0]) * K.T @ (sigma_s - y) + self.lam * torch.sign(self.a)
        return grad

    def fit(self, X_train, y_train, m_epochs=5000, lr=0.001):
        """
        Fits the kernel logistic regression model using gradient descent.

        Args:
            X_train (torch.Tensor): Training features of shape (n, p)
            y_train (torch.Tensor): Training labels of shape (n,)
            m_epochs (int): Number of training epochs
            lr (float): Learning rate for gradient descent :O
        """
        self.X_train = X_train
        n = X_train.shape[0]
        self.a = torch.zeros(n, dtype=torch.float32)

        for epoch in range(m_epochs):
            grad = self.grad(X_train, y_train)
            self.a = self.a - lr * grad

