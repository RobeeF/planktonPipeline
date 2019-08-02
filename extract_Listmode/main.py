# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:35 2019

@author: robin
"""

import os 
from collections import Counter
import pandas as pd
#import numpy as np 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV


# The directory in which you have placed the following code
os.chdir('W:/Bureau/these/planktonPipeline/extract_Listmode')

from from_cytoclus_to_files import extract_features
from from_files_to_clusters import particle_clustering


# Where to look the data at and where to write treated data: Change with yours
data_source = 'W:/Bureau/these/donnees_oceano/old_process_FLR6'
data_destination = 'W:/Bureau/these/data'

seed = 42

#==========================================================================================================================#
#=============================================== Extract the features =====================================================#
#==========================================================================================================================#

extract_features(data_source, data_destination, flr_num = 6)

#==========================================================================================================================#
#========== Make the clusters with k-means from the features and compare it to the true labels ============================#
#==========================================================================================================================#

files_titles = os.listdir(data_destination + '/features')

error_rates, preds_labels = particle_clustering(data_destination + '/features', clus_method = 'k-means')

for date in error_rates.keys():
    print("Sample extracted on ", date)
    print("K-MEANS error is ", error_rates[date])
    print("The following numbers of each particle types were found in the files")
    count = Counter(preds_labels[date]) 
    print(count)
    print('----------------------------------------------------')
     
#==========================================================================================================================#
#============================== Train a supervised RandomForest on the features ===========================================#
#==========================================================================================================================#

# Test with and without SMOTE

# Building the the training and testing sets (The best model will be chosen by crossvalidation)
files_titles = os.listdir(data_destination + '/features')
train = pd.DataFrame()
for i in range(int(len(files_titles)*1/3)):
    df = pd.read_csv(data_destination + '/features/' + files_titles[i], sep = ',', engine = 'python')
    train = df.append(train)

train.set_index(['Particle ID', 'date'], inplace = True)
train = train.dropna(how = 'any')

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
        
## Label Encoding: Turns the labels into numbers
le = LabelEncoder()
le.fit(list(set(Y)))
y = le.transform(Y)

rus = RandomUnderSampler(random_state = seed)
X_train, y_train = rus.fit_sample(X,y)

#sm = SMOTE(random_state = seed)
#X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape %s' % Counter(y_train))

# Finding the best tuning for the Random Forest
rf = RandomForestClassifier(random_state=0, n_jobs = -1)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_

best_rf = RandomForestClassifier(**CV_rfc.best_params_)
best_rf.fit(X_train, y_train)

# Building the the validation set
valid = pd.DataFrame()
for i in range(int(len(files_titles)*2/3), len(files_titles)):
    df = pd.read_csv(data_destination + '/features/' + files_titles[i], sep = ',', engine = 'python')
    valid = df.append(valid)
    
valid.set_index(['Particle ID', 'date'], inplace = True)
valid = valid.dropna(how = 'any')

X_valid = valid.iloc[:, :-1]
y_valid = valid.iloc[:, -1]

# Label Encoding: Turns the labels into numbers
y_valid = le.transform(y_valid)

y_pred_valid = best_rf.predict(X_valid)

# evaluate predictions
accuracy = accuracy_score(y_valid, y_pred_valid)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_valid, y_pred_valid, target_names=le.classes_))



