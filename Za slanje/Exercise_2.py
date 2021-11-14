import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class Softmax():
    def __init__(self, alpha, Niter, batch_size):
        self.alpha = alpha
        self.Niter = Niter
        self.batch_size = batch_size
        pass
    
    def cost(self, X, theta, y):
        X, y = np.asarray(X), np.asarray(y).reshape((-1, 1))
        theta = np.asarray(theta)
        m = X.shape[0]

        first_part = np.zeros((m, 1))
        for i in range(0, m):
            first_part[i] = X[i, : ] @ theta[ : , y[i]]

        second_part = np.exp(X @ theta)
        second_part = np.log( np.sum(second_part, axis = 1) ).reshape((-1, 1))
    
        return np.sum(first_part - second_part)/m

        
    def calculate_phi(self, X, theta, k_class):
        dim = (np.exp(X @ theta[:, k_class])).ndim
        
        if dim == 1:
            t = np.reshape(np.exp(X @ theta[:, k_class]), (-1, 1)) 
        else:
            t = np.exp(X @ theta[:, k_class])

        temp = np.sum(np.exp(X @ theta), axis = 1)
        temp = np.reshape(temp, (temp.shape[0], 1))
        phi_m_x_k = t/temp
        return phi_m_x_k

        
    def delta_J(self, X, theta, y):
        classes = np.arange(theta.shape[1])
        delta_J_n_x_k = np.empty([X.shape[1], 1])
        for k in classes:
            delta_J = np.sum((np.where(y==k, 1, 0) - self.calculate_phi(X, theta, k))*X, axis = 0)/y.shape[0]
            #delta_J = X.T @ (np.where(y==k, 1, 0) - self.calculate_phi(X, theta, k))
            delta_J = np.reshape(delta_J, (-1, 1))
            delta_J_n_x_k = np.append(delta_J_n_x_k, delta_J, axis = 1)
    
        delta_J_n_x_k = delta_J_n_x_k[:, 1:]
        return delta_J_n_x_k
    

    def fit(self, X, y):      
        num_classes = len(np.unique(y))
        theta = np.random.rand(X.shape[1], num_classes)
        theta[:,-1] = 0
        J = []
        
        for it in range(0, self.Niter):
            all_batches = self.make_batches(X, y, self.batch_size)
            for mini_batch in all_batches: 
                X_mini, y_mini = mini_batch
                delta_J_n_x_k = self.delta_J(X_mini, theta, y_mini)
                theta = theta + self.alpha*delta_J_n_x_k
                theta[:,-1] = 0
                J.append(self.cost(X_mini, theta, y_mini)) 
        
        self.J = J
        self.thetas = theta
        return self
    

    def make_batches(self, X, y, size):
        num_batches = int(np.floor(X.shape[0]/size))
        all_batches = []
        
        perm = np.random.permutation(X.shape[0])
        X = X[perm, :]
        y = y[perm]
        
        for i in range(0, num_batches+1):
            if (i+1)*size >= X.shape[0]-1:
                 mini_batch_X = X[i * size:, :] 
                 mini_batch_y = y[i * size:]
                 all_batches.append((mini_batch_X, mini_batch_y))
                 break
                
            mini_batch_X = X[i * size:(i + 1)*size, :] 
            mini_batch_y = y[i * size:(i + 1)*size]
            all_batches.append((mini_batch_X, mini_batch_y))
        return all_batches
    
    def predict(self, X):
        new_y = X @ self.thetas
        y_pred = np.argmax(new_y, axis = 1)
        y_pred = np.reshape(y_pred, (-1,1))
        return y_pred
            
