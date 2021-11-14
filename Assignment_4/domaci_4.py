import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import operator

def rang_features(X, y):
    m, n = X.shape
    all_values = {'Value':[], 'Index':[] } 
    
    for i in range(0,n):
        corr = spearmanr(X[:,i], y)
        all_values['Value'].append(abs(corr[0]))
        all_values['Index'].append(i)
     
    val = all_values['Value'].copy()    
    val.sort(reverse=True)
    all_val_sorted = {'Value': val, 'Index':[]}
    for i in all_val_sorted['Value']:
        ind = all_values['Value'].index(i)
        all_val_sorted['Index'].append(all_values['Index'][ind])
        
    return all_val_sorted['Index']


def forward_selection(X, y):
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 123)
    
    m_train, n = X_train.shape
    m_val, n = X_val.shape
    init_features = np.arange(0, n).tolist() 
    best = []
    
    while 1:
        
        if len(init_features) == 0:
            break
        
        res_features = list(set(init_features) - set(best))
        
        if len(res_features) == 0:
            break
        
        all_values = {'Value':[], 'Index':[] } 
        
        for i in res_features:
            data = X_train[:, best + [i]]
            #data = np.hstack((np.ones((m_train, 1)), data))
            data_val = X_val[:, best + [i]]
            #data_val = np.hstack((np.ones((m_val, 1)), data_val))
            
            model = LogisticRegression(penalty = 'none').fit(data, y_train)
            y_pred = model.predict(data_val)
            scoore = f1_score(y_val, y_pred)
            
            all_values['Value'].append(scoore)
            all_values['Index'].append(i)
            
        
        max_scoore = max(all_values['Value'])
        ind = all_values['Value'].index(max_scoore)
        if max_scoore > 0.7:
            best.append(all_values['Index'][ind])
        else:
            break
        
    return best
            
