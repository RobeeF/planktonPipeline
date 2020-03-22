# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:17:25 2019

@author: Utilisateur
"""

import os
import re
os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from pred_functions import predict
import pandas as pd
from keras.models import load_model
import numpy as np
#from ffnn_functions import scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Model and nomenclature loading
model = load_model('trained_models/LottyNet_FUMSECK')

tn = pd.read_csv('train_nomenclature.csv')
tn.columns = ['Label', 'id']

# Define where to look the data at and where to store preds
#os.chdir('Z:/CS-68-2015/SSLAMM')
export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM_L1"
export_files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files_to_pred = [file for file in export_files if re.search(pulse_regex, file)] # The files containing the data to predict

# Create a log file in the destination folder: list of the already predicted files
preds_store_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM_L2"  # Where to store the predictions
log_path = preds_store_folder + "/pred_logs.txt" # Register where write the already predicted files

if not(os.path.isfile(log_path)):
    open(preds_store_folder + '/pred_logs.txt', 'w+').close()

for file in files_to_pred:
    print('Currently predicting ' + file)
    path = export_folder + '/' + file
    is_already_pred = False
    
    # Check if file has already been predicted
    with open(log_path, "r") as log_file:
        if file in log_file.read(): 
            is_already_pred = True
            
    if not(is_already_pred): # If not, perform the prediction
        # Predict the values
        predict(path, preds_store_folder,  model, tn, is_ground_truth = False)
        
        # Write in the logs that this file is already predicted
        with open(log_path, "a") as log_file:
            log_file.write(file + '\n')
            
    else:
        print(file, 'already predicted')


##############################################################
##################### Local tests to delete ##################
##############################################################
            
##### From basic but full model
model_s = load_model('FUMSECK_trained_scale') # Model trained on scaled and interpolated data
source_path = 'C:/Users/Utilisateur/Desktop/test_pred/L1/FUMSECK-FLR6 2019-04-30 12h36_Default (all)_Pulses.csv'
preds_store_folder = "C:/Users/Utilisateur/Desktop/test_pred/L2"  # Where to store the predictions

predict(source_path, preds_store_folder,  model_s, tn, scale = True, pad = False)


source_path = 'C:/Users/Utilisateur/Desktop/test_pred/L1/SSLAMM_FLR25 2019-12-11 08h08_Default (all)_Pulses.csv'
predict(source_path, preds_store_folder,  model_s, tn, scale = True, pad = False)

# From extracted features (Model13+interp_scaled1510data)
# New data need to be used (Not the case here)
X = np.load('L3_FUMSECK/X_interp_scale.npy')
y = np.argmax(np.load('L3_FUMSECK/y_interp.npy'), axis = 1)

feat_Xor = load_model('Feature_Xor')
XX = feat_Xor.predict(X)

# Train test valid split
#from sklearn.model_selection import StratifiedShuffleSplit
#a = StratifiedShuffleSplit(X, y)
#type(a)
from sklearn.model_selection import train_test_split
XX_train, XX_test, y_train, y_test = train_test_split(XX, y, test_size = 0.33)

rf = RandomForestClassifier(random_state = 0, n_jobs = -1)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator = rf, param_grid = param_grid, cv= 3)
CV_rfc.fit(XX_train, y_train)
CV_rfc.best_params_

best_rf = RandomForestClassifier(**CV_rfc.best_params_)
best_rf.fit(XX_train, y_train)

preds = best_rf.predict(XX_test)
np.mean(preds != y_test)
