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
    
    def hessian(self, X):
        """
        Compute the Hessian matrix for logistic regression.
        """
        s = self.score(X)
        sigma_s = 1 / (1 + torch.exp(-s))
        diag_entries = sigma_s * (1 - sigma_s)
        H = (1 / X.shape[0]) * X.T @ (diag_entries.unsqueeze(1) * X)
        return H

                
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

class NewtonOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha):
        """
        Perform a single Newton update step.

        Arguments:
        X, torch.Tensor: the feature matrix. X_size() == (n,p), 
        where n is the numner of the data points and p is the 
        number of features. This implementation always assumes 
        that the final column of X is  a constant column of 1s. 

        y, torch.tensor: the labels for the Feature matrix X.
        y.size() == (n), where n is the number of data points.

        alpha, float:
        the learning rate of the step.

        """
        grad = self.model.grad(X, y)
        H = self.model.hessian(X)
        reg = 1e-4 * torch.eye(H.shape[0])  #Regularization
        update = torch.linalg.solve(H + reg, grad)
        self.model.w = self.model.w - alpha * update
        return self.model.w



class AdamOptimizer:
    def __init__(self, model):
        self.model = model
        self.m = None
        self.v = None
        self.t = 0

    def step(self, X, y, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, batch_size=32):
        """
        Perform a single epoch of updates using the Adam optimization algorithm.

        Arguments:
        X, torch.Tensor: the feature matrix. X.size() == (n, p), 
        where n is the number of data points and p is the 
        number of features. This implementation always assumes 
        that the final column of X is a constant column of 1s. 

        y, torch.Tensor: the labels for the feature matrix X.
        y.size() == (n,), where n is the number of data points.

        alpha, float:
        the learning rate of the optimizer.

        beta1, float:
        the exponential decay rate for the first moment estimates.

        beta2, float:
        the exponential decay rate for the second moment estimates.

        eps, float:
        a small constant added for numerical stability to avoid
        division by zero. :)

        batch_size, int:
        the size of the minibatch used to compute each gradient step.
"""

        n = X.shape[0]
        indices = torch.randperm(n)
        if self.model.w is None:
            self.model.w = torch.randn(X.shape[1])

        if self.m is None:
            self.m = torch.zeros_like(self.model.w)
            self.v = torch.zeros_like(self.model.w)

        for i in range(0, n, batch_size):
            self.t += 1
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            g = self.model.grad(X_batch, y_batch)

            self.m = beta1 * self.m + (1 - beta1) * g
            self.v = beta2 * self.v + (1 - beta2) * (g ** 2)

            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)

            self.model.w = self.model.w - alpha * m_hat / (torch.sqrt(v_hat) + eps)
        
        return self.model.w
