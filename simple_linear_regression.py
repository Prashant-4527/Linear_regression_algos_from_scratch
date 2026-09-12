import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class SimpleLinearRegression:

    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0

        for i in range(X_train.shape[0]):
           num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean())) 
           den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean())) 
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)
    

    def predict(self, X_test):
        print(X_test)


        return self.m * X_test + self.b


np.random.seed(42)
m_true = 2.5
b_true = -10

CGPA = np.linspace(5, 10, 50)
noise = np.random.normal(0, 0.5, 50)
Package_LPA = m_true * CGPA + b_true + noise

data = {
    "CGPA": CGPA,
    "Package_LPA": Package_LPA
}

df = pd.DataFrame(data)



X = df.iloc[:,0].values
y = df.iloc[:,1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = SimpleLinearRegression()

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

print("Actual:", y_test)
print("Predicted:", predictions)





