import numpy as np

class GDregressor:

    def __init__(self, learning_rate, epochs):
        self.m = 29.19
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs


    def fit(self, X, y):

        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)*X.ravel())

            self.b = self.b - (self.lr * loss_slope_b) 
            self.m = self.m - (self.lr * loss_slope_m)


    def predict(self, X):
        return self.m * X.ravel() + self.b

    
                