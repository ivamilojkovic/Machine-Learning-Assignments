import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class GDA():
    def __init__(self):
        pass

    def predict(self, X):
        m, n = X.shape
        y_pred = []
        
        for i in range(0,m):
            y_try = []
            
            for k in range(0,len(self.p_x_y)):
                y_try.append(self.p_x_y[k].pdf(X[i,:])*self.phi[k])
                         
            y_pred.append(np.argmax(y_try)) 
            
        return y_pred
    
    
    def fit(self, X, y):
        
        m, n = X.shape
        num_class = len(np.unique(y))
    
        p_x_y = []
        phi = []
           
        for k in np.arange(0, num_class):
            
            samples = np.where(y==k)[0]
            
            mean = np.mean((X[samples,:]), axis = 0)
            sigma = np.cov(X[samples,:].T)
            phi.append(np.sum(np.where(y==k, 1, 0))/m)
            
            p_x_y.append(multivariate_normal(mean, sigma))
    
        self.p_x_y = p_x_y
        self.phi = phi
        
        return self  


class GNB():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        
        m, n = X.shape
        num_class = len(np.unique(y))
        p_x_y = []
        phi = []
           
        for k in np.arange(0, num_class):
            samples = np.where(y==k)[0]
            for i in np.arange(0, n):         
                mean = np.mean((X[samples,i]))
                sigma = np.cov(X[samples,i])
                p_x_y.append(norm(mean, sigma))
                
            phi.append(np.sum(np.where(y==k, 1, 0))/m)
         
        p_x_y = np.reshape(p_x_y, (num_class, n))   
                    
        self.p_x_y = p_x_y
        self.phi = phi
        
        return self 
    
    def predict(self, X):
        m, n = X.shape
        y_pred = []
        
        for i in range(0,m):
            all_prob = []
            for k in range(0, len(self.phi)):
                prob = self.phi[k]
                for j in range(0, n):
                    prob *= (self.p_x_y[k, j].pdf(X[i, j]))
                all_prob.append(prob)        
            y_pred.append(np.argmax(all_prob)) 
            
        return y_pred
