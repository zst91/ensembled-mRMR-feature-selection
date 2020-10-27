#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:42:26 2020

@author: Shuaitong Zhang
"""

import random
import numpy as np
import pandas as pd
from skfeature.function.information_theoretical_based import  MRMR

from sklearn.model_selection import train_test_split

def mRMR(X, y, n_selected_features):
    '''
    X, n * d, n cases and d features;
    y, [0, 1]
    n_selected_feature, top n features to select in the importance rank of features
    '''
    feaName = list(X)
    X_ = np.asarray(X)
    
    index_feature, score, muinfo = MRMR.mrmr(X_, y, n_selected_features=n_selected_features)
            
    selected_features = list(map(lambda x: feaName[x], index_feature))
    X_new  = X.iloc[:, index_feature]
    
    return X_new, selected_features

class RobustmRMR:
    def __init__(self, n_iter, n_selected_features=20):
        
        self.index_feature = 0
        self.n_iter = n_iter
        self.n_selected_features = n_selected_features
        
    def fit_transform(self, data, label, n_sample):
        A_train, B_train = data.iloc[np.where(label==0)[0],:], data.iloc[np.where(label==1)[0], :]
        y_A, y_B = np.zeros((n_sample,)), np.ones((n_sample,))
        n_A, n_B = np.shape(A_train)[0], np.shape(B_train)[0]
        
        feature_weight = {}
        feaName = list(data)
        
        for i in feaName:
            feature_weight[i] = 0
    
        for iter_ in range(self.n_iter):
            print(iter_)
            idx_A, idx_B = list(range(n_A)), list(range(n_B))
            
            random.shuffle(idx_A)
            random.shuffle(idx_B)
            
            A_train_ = A_train.iloc[idx_A[:n_sample],:]
            B_train_ = B_train.iloc[idx_B[:n_sample],:]
            
            X_train_ = pd.concat([A_train_, B_train_], axis=0)
            y_train_ = np.append(y_A, y_B)
            
            # perform mrmr in the resampled dataset X_train_
            _, selName = mRMR(X_train_, y_train_, n_selected_features=20)
#            _, selName = fs_sel.fit_transform(X_train_, y_train_)
            
            for i in selName:
                feature_weight[i] += 1
            
        self.index_feature = np.argsort(list(feature_weight.values()))[::-1][:self.n_selected_features]
        
        selected_feature = list(map(lambda x: feaName[x], self.index_feature))
        data_new = data.iloc[:, self.index_feature]
        return data_new, selected_feature
    
    def get_support(self):
        return self.index_feature

if __name__ == "__main__":
    
    #TODO: read your own feature file
    X_train = pd.read_csv('/trainfeature.csv')
    X_test  = pd.read_csv('/testfeature,csv')
    y_train = 'xxx'
    y_test  = 'xxx'
    
    robustmrmr = RobustmRMR(n_iter=250, n_selected_features=5)
    X_new, features_selected = robustmrmr.fit_transform(X_train, y_train, n_sample=50)
    
    X_test_new = X_test.iloc[:, robustmrmr.get_support()]
    
        

                
