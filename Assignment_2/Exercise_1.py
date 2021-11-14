import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from matplotlib import pyplot as plt

class Logistic_Regression():
    def __init__(self, alpha, Niter = 1000, tol = 0.001):
        self.alpha = alpha
        self.Niter = Niter
        self.tol = tol
    
    def cost(self, theta, X, y):
        h = self.sigmoid(X.dot(theta))
        J = y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))
        return J 

    def fit(self, X, y):        
        m, n = X.shape
        theta = np.ones((n, 1))
        y = np.reshape(y, (m, 1))
        
        all_cost = []
        all_theta = []
        for it in np.arange(0, self.Niter):
            all_theta.append(theta)
            all_cost.append(self.cost(theta, X, y))
            
            delta_J = 1/m * (X.T).dot(self.sigmoid(X.dot(theta)) - y)
            theta = theta - self.alpha*delta_J
            if abs(all_cost[-1] - self.cost(theta, X, y)) <= self.tol:
                #print('Prekinulo se u ', str(it), ". iteraciji!")
                break
            #if it == self.Niter-1:
                #print('Proslo je kroz sve iteracije!')
                
        self.all_cost = all_cost
        self.all_theta = all_theta
        self.last_theta = all_theta[-1]
        
        return self
        
    def predict(self, X, theta):
        y = X.dot(theta)
        y_pred = np.argmax(y, axis = 1)
        return y_pred
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
 
def make_new_y(y):
    m, n = y.shape
    classes = np.unique(y)

    y_all = np.empty([m, 1])
    for k in classes:
        y_all = np.append(y_all, np.where(y==k, 1, 0), axis = 1)
    y_all = y_all[:, 1:]
    return y_all

