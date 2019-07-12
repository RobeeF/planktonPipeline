# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:15:30 2019

@author: robin
"""

import matplotlib.pyplot as plt 
import pandas as pd
import os 
import re
import numpy as np

#============================================================================
# Pulse image reconstruction
#============================================================================

def extract_imgs(data_source, data_destination, flr_num = 6, max_cc_instance_per_date = 10, noise_pcles_to_load =  10 ** 4):
    ''' Create 5 images per particle of the files in the data_source repository. The images are placed in folders mentioning the label of the particle
    data_source (str): The path to the original source of data
    data_features (str) : The path to write the data to
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    max_cc_instance_per_date (int): Maximum number of curves (of each of the 5 types) to extract for each date. Avoid to generate a very big amount of curves
    noise_pcles_to_load (int): As there are a very high number of particles per date, it is not useful to extract them all
    ---------------------------------------------------------------------------------------------
    returns (None): Write the images on hard disk
    '''
    assert (flr_num == 6) or (flr_num == 25)
    
    files_title = [f for f in os.listdir(data_source)]
    # Keep only the interesting csv files
    flr_title = [f for f in files_title if re.search("^FUMSECK-FLR" + str(flr_num),f) and re.search("csv",f) ]

    pulse_titles_clus = [f for f in flr_title if  re.search("Pulse",f) and not(re.search("Default",f))]
    pulse_titles_default = [f for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)]

    dates = set([re.search("FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z]+" , f).group(1) for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)])
    cluster_classes = list(set([re.search("_([a-z]+)_Pulses.csv", cc).group(1) for cc in pulse_titles_clus]))
    cluster_classes += ['noise']

    curves_types = ['Curvature','FWS','SWS','FL Orange','FL Red']
    
    ### Create the directories that will host the curves
    if not os.path.exists(data_destination + '/curves'):
        os.makedirs(data_destination + '/curves') 
        os.makedirs(data_destination + '/curves/all')  
        os.makedirs(data_destination + '/curves/train')
        os.makedirs(data_destination + '/curves/test')   
        os.makedirs(data_destination + '/curves/valid')
        
        for ctype in curves_types:
            for cclass in cluster_classes:
                os.makedirs(data_destination + '/curves/all/' + ctype + '/' + cclass)  
                os.makedirs(data_destination + '/curves/train/' + ctype + '/' + cclass) 
                os.makedirs(data_destination + '/curves/test/' + ctype + '/' + cclass)
                os.makedirs(data_destination + '/curves/valid/' + ctype + '/' + cclass)
    
    # Populate only the 'all' directory. The split between the 3 other repos will be undertaken by another function
    directory_to_populate = data_destination + '/curves/all'
    
    for date in dates: # For each sampling phase
        print("Processing:", date)
        pulse_data = pd.DataFrame()
        # Get the info about each particule and its cluster label
        date_datasets_titles = [t for t in pulse_titles_clus if re.search(date, t)]
        
        ### Extract the curves of non-noise particules (cryprophytes etc...)
        # For each file
        for title in date_datasets_titles: 
            print(title)
            count = 0
            
            df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
            df = df[df["Particle ID"] != 0] # Delete formatting zeros
    
            # Get the name of the cluster from the file name
            clus_name = re.search("_([a-z]+)_Pulses.csv", title).group(1) 
                        
            # Generation of the 5 curves for each particle of each cluster      
            for idx, group in df.groupby(["Particle ID"]):
                if count >= max_cc_instance_per_date: # Enable to generate a finite amount of data
                    break
                if idx != 0:
                    for col in group.columns[1:]:
                        plt.figure(figsize=(3,2)) 
                        plt.plot(group[col].tolist())
                        plt.axis('off')
                        plt.savefig(directory_to_populate + '/' + col + '/' + clus_name + '/' + date + '_' + str(count) + '.png', \
                                    format='png', dpi=100)
                        plt.close()
                    count += 1
                    
    
            # Add the date of the extraction
            df["date"] = date
            df = df[['Particle ID', 'date']] # The other 5 columns have already been used to generate curves: no need to keep then
            df.set_index("Particle ID", inplace = True)
            pulse_data = pulse_data.append(df)
            
        clus_indices = set(pulse_data.index)
        #len_pulse_data = len(pulse_data)
        del(pulse_data) # Free the memory to avoid Memory Errors
                
        ### Extract the curves of noise particules
        print("Extracting noise particles")
        title = [t for t in pulse_titles_default if re.search(date, t)][0] # Dirty
        df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)#, nrows = noise_pcles_to_load) 
        df.set_index("Particle ID", inplace = True)
        
        # Get the indices of the noise particles to generate the associated curves
        all_indices = set(df.index)
        noise_indices = list(all_indices - set(clus_indices))
        noise = df.loc[noise_indices] # Keep only the noise particles 
        clus_name = "noise"
        
        count = 0
        # Generation of the 5 curves for each noise particle     
        for idx, group in df.groupby(["Particle ID"]):
            if count >= max_cc_instance_per_date: # Avoid to have too much data 
                break
            if idx != 0:
                for col in group.columns:
                    plt.figure(figsize=(3,2)) 
                    plt.plot(group[col].tolist())
                    plt.axis('off')
                    plt.savefig(directory_to_populate + '/' + col + '/' + clus_name + '/' + date + '_' + str(count) + '.png', \
                                    format='png', dpi=100)
                    plt.close()
                count += 1
                
            
            # Sanity check 
            #if (len(noise) + len_pulse_data != len(df)):
                #raise ValueError("It seems that some data have been lost during extraction")
                
        # Free useless memory
        del(df)
        del(noise)
                
    
