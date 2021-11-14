import numpy as np
from scipy.optimize import fmin_powell
import pandas as pd
from scipy.optimize import least_squares

class Linear_Regression():
    
    def __init__(self):
        pass
        
    def fit(self, X, y):
        theta_opt = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.theta = theta_opt
        return self
    
    def get_theta(self):
        return self.theta
    
    def predict(self, X_test):
        pred = X_test.dot(self.theta)
        return pred
    
class Polynomial(Linear_Regression):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        # X_new =  self.make_poly(X, self.degree)
        theta_opt = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.theta = theta_opt
        return self
     
    def make_poly(self, X, degree):
        self.degree = degree
        m, n = X.shape
        if degree!=1:
            for i in np.arange(degree,-1,-1):
                for j in np.arange(degree-i,-1,-1):
                    for k in np.arange(degree-j-i,-1,-1):
                        for l in np.arange(degree-k-j-i,-1,-1):                           
                            new_feature = X[:,1]**i * X[:,2]**j * X[:,3]**k * X[:,4]**l * X[:,5]**(degree-i-j-k-l) 
                            #print(i,j,k,l,degree-i-j-k-l)
                            X = np.concatenate((X, new_feature.reshape(m,1)), axis=1)
        return X 
    
    def predict(self, X_test):
        # X_test = self.make_poly(X_test, self.degree)
        pred = X_test.dot(self.theta)
        return pred

    
class Ridge(Linear_Regression):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y, lam):
        m, n = X.shape       
        I = np.eye(n)
        opt_theta = np.linalg.inv(X.T.dot(X) + lam*I).dot(X.T).dot(y)
        self.theta =  opt_theta
        return self
    
class Lasso(Linear_Regression):
    
    def __init__(self):
       super().__init__()
    
    def fit(self, X, y, lam):
        m, n = X.shape
        self.init_theta = np.ones(n)
        
        theta_opt = fmin_powell(self.cost, x0 = self.init_theta, 
                                args = (X, y, lam), disp = False)
        self.theta = theta_opt
        return self
    
    def cost(self, x, X, y, lam):
        m, n = X.shape
        J = sum((y-X.dot(x))**2) + lam*sum(abs(x[1:]))
        return J
        
class Locally_Weighted(Linear_Regression):
    
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y):
        all_theta = []
        
        for W in self.weights:
            theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
            all_theta.append(theta)
        
        self.thetas = all_theta
        return self
        
    def predict(self, X):
        m, n = X.shape
        y_pred = np.zeros((m,))
        
        for i, theta in enumerate(self.thetas):
            y_pred[i,] = X[i,:].dot(theta) 

        return y_pred
    
    def weight_matrices(self, X_train, X_test, tau):
        self.test = X_test
        
        k, n = X_train.shape
        l, n = X_test.shape
        
        I = np.eye(k)
        
        W_s = [] 
        for j in range(0,l):
            # W = np.exp(-(X_train - X_test[j,:]).dot((X_train - X_test[j,:]).T)/(2*tau**2))
            W = np.exp(-np.sum((X_train - X_test[j,:])**2, axis = 1)/(2*tau**2))
            W = W.reshape((-1,))          
            W_s.append(np.diag(W))
           
        self.weights = W_s
        return self
    
    def get_theta(self):
        return self.thetas
    
    
def split_data(X, y, p):
    m = len(y)
    arr = np.arange(m)
    arr_per = np.random.permutation(arr)
    
    ntest = int(p/100*m)
    
    X_test = X[arr_per[0:ntest],:]
    X_train = X[arr_per[ntest:-1],:]
    y_test = y[arr_per[0:ntest]]
    y_train = y[arr_per[ntest:-1]]
    
    return X_train, X_test, y_train, y_test    
    
def MSE_metric(y_pred, y):
    m = len(y) 
    return (1/m)*sum((y_pred-y)**2)   


