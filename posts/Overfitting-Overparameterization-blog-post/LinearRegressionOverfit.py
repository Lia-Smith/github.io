import torch
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
            self.w = torch.rand((X.size()[1]), dtype=X.dtype)


        # your computation here: compute the vector of scores s
        s = X @ self.w
        return s
    
class MyLinearRegression(LinearModel):

    def predict(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of score is score[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            score torch.Tensor: vector of scores. score.size() = (n,)
        """
        score = self.score(X)
        return score
    
    def loss(self, X, y):
        """
        Compute the mean squared error for each point in the feature matrix X. 
        The formula for the ith entry of  loss is loss[i] = (X*w-y)^2 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, toch.tensor: the labels vector. y.size() == (n,1),
            where n is the number of data points. This implementation
            assumes that y is a continous label rather than a discrete one.

        RETURNS: 
            mse torch.Tensor: vector of losses. mse.size() = (n,)
        """
        mse = (self.predict(X) - y)**2 #mean squared error
        return mse.mean()
    
class OverParameterizedLinearRegressionOptimizer:
    def __init__(self, model):
        self.model = model


    def fit(self,X,y):
        w_star = torch.linalg.pinv(X) @ y
        self.model.w = w_star

def sig(x): 
    return 1/(1+torch.exp(-x))

def square(x): 
    return x**2

class RandomFeatures:
    """
    Random sigmoidal feature map. This feature map must be "fit" before use, like this: 

    phi = RandomFeatures(n_features = 10)
    phi.fit(X_train)
    X_train_phi = phi.transform(X_train)
    X_test_phi = phi.transform(X_test)

    model.fit(X_train_phi, y_train)
    model.score(X_test_phi, y_test)

    It is important to fit the feature map once on the training set and zero times on the test set. 
    """

    def __init__(self, n_features, activation = sig):
        self.n_features = n_features
        self.u = None
        self.b = None
        self.activation = activation

    def fit(self, X):
        self.u = torch.randn((X.size()[1], self.n_features), dtype = torch.float64)
        self.b = torch.rand((self.n_features), dtype = torch.float64) 

    def transform(self, X):
        return self.activation(X @ self.u + self.b)