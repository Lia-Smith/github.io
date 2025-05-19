import torch


#  f (w)= objective_function = min( lambda / 2 ||w||^2 + 1/m * sum( loss(w;(x_i, y_i) ) )
# loss(w;(x_i, y_i)) = max (0, 1 - y <w_i,x_i>)
# consider how much worse a false negative is than a false positive (want more false positives)

#
# 
# 
# Trying to add mini-batch gradient descent to the SVM
class SVM: 
    
    def __init__(self, n_iters = 1000, lam  = 0.0001, batch_size = 32): # n_iters = T
        self.n_iters = n_iters
        self.lam = lam
        self.batch_size = batch_size
        self.w = None

    def score(self, X):
        if self.w is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.w
    
    def fit(self, X, y): #Check if of y should be 1 or -1
        y_ = torch.where(y <= 0, -1, 1) # set y to -1 or 1
        self.w = torch.zeros(X.shape[1]) # initalize weights
        s = X.shape[0]
        for t in range(1, self.n_iters): 
            A = torch.randint(0, s, (self.batch_size,))
            Ap = []
            ap_sum = 0
            nu = 1 / (self.lam * t)
            for i in A:
                x_i = X[i]
                y_i = y_[i]
                if (y_i * self.score(x_i) < 1):
                    Ap.append(i.item())
                    ap_sum += y_i * x_i
                else: 
                    continue

            self.w = (1 - nu * self.lam) * self.w  + (nu / self.batch_size) * ap_sum

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model is not fitted yet.")
        score = self.score(X) >= 0
        return score


















# class SVM: 
    
#     def __init__(self, n_iters = 1000, lam  = 0.0001): # n_iters = T
#         self.n_iters = n_iters
#         self.lam = lam
#         self.w = None

#     def score(self, X):
#         if self.w is None:
#             raise ValueError("Model is not fitted yet.")
#         return X @ self.w
    
#     def fit(self, X, y): #Check if of y should be 1 or -1
#         y_ = torch.where(y <= 0, -1, 1) # set y to -1 or 1
#         self.w = torch.zeros(X.shape[1]) # initalize weights
#         s = X.shape[0]
#         for t in range(1, self.n_iters): 
#             i = torch.randint(0, s, ()).item()
#             x_i = X[i]
#             y_i = y_[i]
#             nu = 1 / (self.lam * t)
#             if y_i  * self.score(x_i) < 1:
#                 self.w = (1 - nu * self.lam) * self.w + nu* y_i * x_i
#             else: 
#                 self.w = (1 - nu * self.lam) * self.w

#     def predict(self, X):
#         if self.w is None:
#             raise ValueError("Model is not fitted yet.")
#         score = self.score(X) >= 0
#         # bin_score = score.float()
#         # final = torch.where(bin_score == 1, 1, -1)
#         return score





























# This is the basic Pegasos SVM implementation

# import torch

# class SVM: 
#     def __init__(self, n_iters=10000, lam=0.01):
#         self.n_iters = n_iters
#         self.lam = lam
#         self.w = None

#     def score(self, X):
#         if self.w is None:
#             raise ValueError("Model is not fitted yet.")
#         return X @ self.w

#     def fit(self, X, y):
#         if not torch.all((y == 1) | (y == -1)):
#             raise ValueError("Labels must be -1 or 1.")
#         self.w = torch.zeros(X.shape[1])
#         s = X.shape[0]
#         for t in range(1, self.n_iters + 1):
#             i = torch.randint(0, s, ()).item()
#             x_i = X[i]
#             y_i = y[i]
#             nu = 1 / (self.lam * t)
#             if y_i * self.score(x_i) < 1:
#                 self.w = (1 - nu * self.lam) * self.w + nu * y_i * x_i
#             else:
#                 self.w = (1 - nu * self.lam) * self.w

#             # Project weights onto ball
#             # norm = torch.norm(self.w)
#             # scale = min(1.0, 1.0 / (norm * (self.lam ** 0.5)))
#             # self.w = self.w * scale

#     def predict(self, X):
#         return self.score(X) >= 0