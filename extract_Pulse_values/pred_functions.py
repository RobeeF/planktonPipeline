# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:24:12 2019

@author: Robin
"""

import os

os.chdir('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from ffnn_functions import custom_pad_sequences, interp_sequences
import pandas as pd
import numpy as np
import re
import scipy.integrate as it
from ffnn_functions import scaler


def predict(source_path, dest_folder, model, tn, scale = False, pad = False):
    ''' Predict the class of unlabelled data with a pre-trained model and store them in a folder
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    model (ML model): the pre-trained model to use, in order to make predictions
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a csv on harddisk directly 
        '''
  
    max_len = 120 # The standard length to which is sequence will be broadcasted

    #with open(source_path, 'r') as txt:
    file_id_pattern = "(FLR[0-9]{1,2} 20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"
    file_id = re.search(file_id_pattern, source_path).group(1)
        
    #csv_like_data = re.sub(pattern5, r'\2.\3', csv_like_data) # Neglect some commas
    # Indent break
    # Convert as dataframe
    df = pd.read_csv(source_path, sep = ';', decimal = ',')
    assert(len(df[df.isna().any(axis = 1)]) == 0)
    
    df[df.isna().any(axis = 1)]['Particle ID'].value_counts()
    #df = df.dropna(how = 'any',axis = 0) # Dirty drop due to formating issues
    df['Curvature'].astype(float)
     
    # Reformat the data. Each obs shape is (nb_curves, seq_len)
    obs_list = []
    total_fws_list = []
    total_sws_list = []
    total_flo_list = []
    total_flr_list = []
    total_curv_list = []
    
    for pid, obs in df.groupby('Particle ID'):
        obs_list.append(obs.iloc[:,:5].values.T)
        total_fws_list.append(it.trapz(obs['FWS'].astype(float)))
        total_sws_list.append(it.trapz(obs['SWS'].astype(float)))
        total_flo_list.append(it.trapz(obs['FL Orange'].astype(float)))
        total_flr_list.append(it.trapz(obs['FL Red'].astype(float)))
        total_curv_list.append(it.trapz(obs['Curvature'].astype(float)))
        
    # Defining a fixed length for all the sequence: 0s are added for shorter sequences and longer sequences are truncated    
    if pad:
        obs_list = custom_pad_sequences(obs_list, max_len)
    else:
        obs_list = interp_sequences(obs_list, max_len)
            
    X = np.transpose(obs_list, (0, 2, 1))
    if scale:
        X = scaler(X)
        
    preds = np.argmax(model.predict(X), axis = 1)
        
    formatted_preds = pd.DataFrame({'Particle': list(set(df['Particle ID'])), 'Particle_class_num': preds, \
                                    'Total FWS': total_fws_list, 'Total SWS': total_sws_list, \
                                    'Total FLO': total_flo_list, 'Total FLR': total_flr_list, 
                                    'Total CURV': total_curv_list}) # Pourquoi list(set(Particle ID)) ?
    
    # Add string labels and store the predictions on hard disk 
    formatted_preds = formatted_preds.merge(tn)
    formatted_preds.to_csv(dest_folder + '/' + file_id + '.csv', index = False)
