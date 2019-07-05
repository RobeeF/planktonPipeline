# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:17:41 2019

@author: robin
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd 
import numpy as np
import os 

def least_permut_error(true_encoded_label, pred_encoded_label):
    ''' Try every encoding for the predictions to determine the least possible error
    true_encoded_label (array-like): The encoded manual labels 
    pred_encoded_label (array-like) : The encoded labels determined by the algorithm
    Example: (to add):
    ---------------------------------------------------------------------------------------
    returns (float, array-like): The error rate commited on the dataset and the rightly encoded label predictions
    '''
    min_error = np.inf
    nb_labels = len(set(true_encoded_label))
    n  = len(true_encoded_label)
    right_encoded_preds = []
    
    for i in range(nb_labels):
        new_labels = (pred_encoded_label + i) % nb_labels
        
        error = np.sum(true_encoded_label != new_labels)/n
        if (min_error >= error):
            right_encoded_preds  = new_labels
            min_error = error
    return min_error, right_encoded_preds


def particle_clustering(files_dir, clus_method = 'k-means'):
    files_title = [f for f in os.listdir(files_dir)]
    df = pd.DataFrame()
    error_rates = {}
    preds_labels = {}
    
    for title in files_title:
        print(title)
        
        df = pd.read_csv(files_dir + '/' + title)
        date = df['date'][0]
        
        df.set_index(['Particle ID', 'date'], inplace = True)
        df = df.dropna(how = 'any')

        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
        
        # Label Encoding: Turns the labels into numbers
        le = LabelEncoder()
        le.fit(list(set(Y)))
        Y_num = le.transform(Y)
        
        true_clus_nb = len(set(Y))
        
        # Kmeans fit 
        if clus_method == 'k-means':
            kmeans = KMeans(n_clusters=true_clus_nb, random_state=0).fit(X) # Fit with the right number of clusters
            error_rate, preds_num = least_permut_error(Y_num, kmeans.labels_)
            y_pred_label = le.inverse_transform(preds_num)
            
            error_rates[date] = error_rate
            preds_labels[date] = y_pred_label

        else:
            raise RuntimeError("The requested method is not implemented yet.")
    return error_rates, preds_labels

    
    
