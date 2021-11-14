import numpy as np
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt


class SVM():
    
    def __init__(self, kernel = 'linear', C = None, degree = 1, gamma = 0.5):
        
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.gamma = gamma
        
    def fit(self, X, y):
        
        m, n = X.shape   
        
        if self.C == None:
            G = matrix(-1*np.eye(m))
            h = matrix(np.zeros(m))
        else:
            g1 = -1*np.eye(m)
            g2 = np.eye(m)
            G = matrix(np.vstack((g1, g2)))
            h1 = np.zeros(m)
            h2 = self.C * np.ones(m)
            h = matrix(np.hstack((h1, h2)))
            
        q = matrix(-1*np.ones((m, 1)))
        A = matrix(y, (1, m))
        b = matrix(0.0)
    
        H = np.zeros((m,m))
        K = np.zeros((m,m))
        
        for i in range(0, m):
            for j in range(0, m):
                
                if self.kernel  == 'linear':
                    K[i, j] = linear_K(X[i,:], X[j,:])
                if self.kernel == 'polynomial':
                    K[i, j] = polynomial_K(X[i,:], X[j,:], self.C, self.degree)
                if self.kernel == 'gaussian':
                    K[i, j] = Gaussian_K(X[i,:], X[j,:], self.gamma)
                    
                H[i,j] = y[i] * y[j] * K[i, j]
                
        P = matrix(H)        
                
        solvers.options['show_progress'] = False        
        solver = solvers.qp(P, q, G, h, A, b)
        
        alpha_opt = np.ravel(solver['x'])
        self.alphas = alpha_opt
        
        # Support vectors are those vectors that have positive alpha
        sv_bool = alpha_opt > 0 
        sv_ind = np.where(alpha_opt > 0)[0]
        support_vectors_alpha = alpha_opt[sv_bool].reshape(-1, 1)
        support_vectors_y = y[sv_bool]
        support_vectors_X = X[sv_bool]       
        self.support_vectors = (support_vectors_X, support_vectors_y, support_vectors_alpha)
        self.sv_index = sv_ind

        
        # Number of samples for making Kernel and calculate the intercept b
        if self.C != None:
            sample_ind = np.where(np.logical_and(alpha_opt > 0, alpha_opt < self.C))[0]
        else:
            sample_ind = np.where(alpha_opt > 0)[0]
               
        # Separation line's weights
        w_opt = np.sum(support_vectors_alpha * support_vectors_y * support_vectors_X, axis = 0).reshape(-1,1)
        self.w = w_opt
        
        # Separation line's intercept
        b_opt = 0
        for i, s in enumerate(sample_ind):
            b_opt += 1/y[s]
            for j, k in enumerate(sv_ind):
                b_opt -=  support_vectors_alpha[j] * support_vectors_y[j] * K[s, k]
        
        self.b = b_opt/len(sample_ind)
        
        # Separation line 
        self.decision_boundry = self.compute_f(X) 
             
        return self
    
    def compute_f(self, X):
        m, n = X.shape
        
        if self.kernel=='linear':
            if self.C==None:
                f = X @ self.w
            else:
                f = X @ self.w - self.C
                
        else:
            sv_alpha = self.support_vectors[2]           
            sv_X = self.support_vectors[0]            
            sv_y = self.support_vectors[1]
            
            
            f = np.zeros((m, 1))
            for j in range(0, m):               
                t = 0
                for i in range(0, len(self.sv_index)):
                    
                    if self.kernel == 'gaussian':
                        t += sv_alpha[i] * sv_y[i] * Gaussian_K(X[j,:], sv_X[i,:], self.gamma)
                        
                    if self.kernel == 'polynomial':
                        t += sv_alpha[i] * sv_y[i] * polynomial_K(X[j,:], sv_X[i,:], self.C, self.degree)
                    
                f[j] = t
                
        f += self.b                
        return f
    
    def predict(self, X):
        return np.sign(self.compute_f(X))
                
        

def linear_K(X1, X2):
    K = X1 @ X2.T
    return K

def polynomial_K(X1, X2, C = 1, degree = 2):
    if C==None:
        K = (1 + X1 @ X2.T)**degree
    else:
        K = (C + X1 @ X2.T)**degree
    return K

def Gaussian_K(X1, X2, gamma = 0.5):
    K = np.exp(-np.linalg.norm(X1-X2)**2 / gamma**2 /2)
    return K

    