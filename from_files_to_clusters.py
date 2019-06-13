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
from collections import Counter

files_loc = 'C:/Users/robin/Documents/Documents_importants/scolarité/These/pipeline/data/features'
os.chdir(files_loc)


def least_permut_error(true_encoded_label, pred_encoded_label):
    ''' Try every encoding for the predictions to determine the least possible error'''
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

files_title = [f for f in os.listdir('.') if os.path.isfile(f)]

df = pd.DataFrame()

for title in files_title:
    df = df.append(pd.read_csv(title))
    break

df.set_index(['Particle ID', 'date'], inplace = True)
df = df.dropna(how = 'any')
# train_test_split
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X.columns
X.iloc[:,0]

# Label Encoding
le = LabelEncoder()
le.fit(list(set(Y)))
Y_num = le.transform(Y)

true_clus_nb = len(set(Y))

# Kmeans fit 
kmeans = KMeans(n_clusters=true_clus_nb, random_state=0).fit(X) # Fit with the right number of clusters
error_rate, preds_num = least_permut_error(Y_num, kmeans.labels_)
print("K-MEANS error is ", error_rate)

preds_labels = le.inverse_transform(preds_num)
count = Counter(preds_labels)

# DEC ? 
