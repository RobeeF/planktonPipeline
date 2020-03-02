# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:24:12 2019

@author: Robin
"""

from ffnn_functions import custom_pad_sequences, interp_sequences, homogeneous_cluster_names
import pandas as pd
import numpy as np
import scipy.integrate as it
from ffnn_functions import scaler
import fastparquet as fp
import re
import matplotlib.pyplot as plt

def predict(source_path, dest_folder, model, tn, scale = False, pad = False):
    ''' Predict the class of unlabelled data with a pre-trained model and store them in a folder
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    model (ML model): the pre-trained model to use, in order to make predictions
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a csv on hardisk directly 
        '''
    
    max_len = 120 # The standard length to which is sequence will be broadcasted

    pfile = fp.ParquetFile(source_path)
    df = pfile.to_pandas()

    # Reformat the data. Each obs shape is (nb_curves, seq_len)
    obs_list = []
    pid_list = []

    total_fws_list = []
    total_sws_list = []
    total_flo_list = []
    total_flr_list = []
    total_curv_list = []
    true_labels = []
    
    try:
        df = df.set_index('Particle ID')
    except:
        print('Particle ID was not found in column names')
        
    for pid, obs in df.groupby('Particle ID'):
        pid_list.append(pid)
        obs_list.append(obs.iloc[:,:5].values.T) 
        total_fws_list.append(it.trapz(obs['FWS'].astype(float)))
        total_sws_list.append(it.trapz(obs['SWS'].astype(float)))
        total_flo_list.append(it.trapz(obs['FL Orange'].astype(float)))
        total_flr_list.append(it.trapz(obs['FL Red'].astype(float)))
        total_curv_list.append(it.trapz(obs['Curvature'].astype(float)))
        true_labels.append(np.unique(obs['cluster'])[0])
        
        
    # Defining a fixed length for all the sequence: 0s are added for shorter sequences and longer sequences are truncated    
    if pad:
        obs_list = custom_pad_sequences(obs_list, max_len)
    else:
        obs_list = interp_sequences(obs_list, max_len)
    
    X = np.transpose(obs_list, (0, 2, 1))
    if scale:
        X = scaler(X)
        
    preds = np.argmax(model.predict(X), axis = 1)
    
    true_labels = homogeneous_cluster_names(np.array(true_labels))
    formatted_preds = pd.DataFrame({'Particle ID': pid, \
                                    'Total FWS': total_fws_list, 'Total SWS': total_sws_list, \
                                    'Total FLO': total_flo_list, 'Total FLR': total_flr_list, \
                                    'Total CURV': total_curv_list, \
                                    'True FFT id': None, 'True FFT Label': true_labels, \
                                    'Pred FFT id': preds, 'Pred FFT Label': None}) 
    
    # Add string labels
    tn_dict = tn.set_index('id')['Label'].to_dict()
    
    for id_, label in tn_dict.items():
        formatted_preds.loc[formatted_preds['True FFT Label'] == label, 'True FFT id'] = id_
        formatted_preds.loc[formatted_preds['Pred FFT id'] == id_, 'Pred FFT Label'] = label

    # Store the predictions on hard disk 
    date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
    file_name = re.search(date_regex, source_path).group(1)
    formatted_preds.to_csv(dest_folder + '/' + file_name + '.csv', index = False)


def plot_2D(preds, tn, q1, q2, loc = 'upper left'):
    ''' Plot 2D cytograms as for manual classification '''
    
    colors = ['#96ceb4', '#ffeead', '#ffcc5c', '#ff6f69', '#588c7e', '#f2e394', '#f2ae72', '#d96459']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    for id_, label in enumerate(list(tn['Label'])):
        obs = preds[preds['True FFT Label'] == label]
        ax1.scatter(obs[q1], obs[q2], c = colors[id_], label= label)
        ax1.legend(loc= loc, shadow=True, fancybox=True, prop={'size':8})
    
    ax1.set_title('True :' +  q1 + ' vs ' + q2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(q1)
    ax1.set_ylabel(q2)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(1, 10**6)
    
    
    for id_, label in enumerate(list(tn['Label'])):
        obs = preds[preds['Pred FFT Label'] == label]
        ax2.scatter(obs[q1], obs[q2], c = colors[id_], label= label)
        ax2.legend(loc= loc, shadow=True, fancybox=True, prop={'size':8})
    ax2.set_title('Pred :' +  q1 + ' vs ' + q2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(q1)
    ax2.set_ylabel(q2)
    ax2.set_xlim(1, 10**6)
    ax2.set_ylim(1, 10**6)
    
