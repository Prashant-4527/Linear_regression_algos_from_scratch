import numpy as np

class GDregressor:

    def __init__(self, learning_rate, epochs):
        self.m = 29.19
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs


    def fit(self, X, y):

        for i in range(self.epochs):
            loss_slope = -2 * np.sum(y - self,m*X.ravel() - self.b)   
            self.b = self.b - (self.lr * loss_slope) 
        print(self.b)
                