# How would i even start something like this...
#Takes in a feature vector X and a label vector y and performs gradient boosting on it 
import torch 
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradientBoostClassifier():
    def __init__(self, num_iterations, learning_rate, max_depth):
        self.log_odds = None
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def fit(self, X, y):
        X_np = X.numpy()
        y_np = y.numpy().astype(float).squeeze()

        prob_pneumonia = np.mean(y_np)
        self.log_odds = torch.tensor(np.log(prob_pneumonia/(1-prob_pneumonia)))
        preds = torch.squeeze(torch.full_like(y, self.log_odds))
    
        for _ in range(self.num_iterations): #Create regression tree weak learners, 
            pseudo_residuals = torch.squeeze(y) - self.sigmoid(preds)
            tree = DecisionTreeRegressor(max_depth=self.max_depth) 
            tree.fit(X_np, pseudo_residuals.numpy())
            self.trees.append(tree)
            preds += self.learning_rate * torch.tensor(tree.predict(X_np))

    def predict(self, X):
        preds = torch.full((X.shape[0],), self.log_odds)
        for tree in self.trees:
            preds += self.learning_rate * torch.tensor(tree.predict(X.numpy()))
        return (torch.sigmoid(preds) > 0.5).float()
