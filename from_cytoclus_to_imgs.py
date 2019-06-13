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

data_source = 'C:/Users/robin/Documents/Documents_importants/scolarité/These/ROBIN'
data_destination = 'C:/Users/robin/Documents/Documents_importants/scolarité/These/pipeline/data'
os.chdir(data_source)

#============================================================================
# Pulse image reconstruction
#============================================================================

files_title = [f for f in os.listdir('.') if os.path.isfile(f)]
# Keep only the interesting csv files
flr6_title = [f for f in files_title if re.search("^FUMSECK-FLR6",f) and re.search("csv",f) ]

pulse_titles_clus = [f for f in flr6_title if  re.search("Pulse",f) and not(re.search("Default",f))]
pulse_titles_default = [f for f in flr6_title if  re.search("Pulse",f) and re.search("Default",f)]

dates = set([re.search("FLR6 (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z]+" , f).group(1) for f in flr6_title if  re.search("Pulse",f) and re.search("Default",f)])
cluster_classes = list(set([re.search("_([a-z]+)_Pulses.csv", cc).group(1) for cc in pulse_titles_clus]))
cluster_classes += ['noise']

# create the directories that will host the curves
if not os.path.exists(data_destination + '/curves'):
    os.makedirs(data_destination + '/curves') 
    os.makedirs(data_destination + '/curves/all')   
    os.makedirs(data_destination + '/curves/train')   
    os.makedirs(data_destination + '/curves/validation')

directory_to_populate = data_destination + '/curves/all'

# Maximum number of particles of each class to process for each date. 
max_cc_instance_per_date = 10 # Set to np.inf to process all the particles

for date in dates: # For each sampling phase
    print(date)
    pulse_data = pd.DataFrame()
    # Get the info about each particule and its cluster label
    date_datasets_titles = [t for t in pulse_titles_clus if re.search(date, t)]
    for title in date_datasets_titles: 
        count = 0
        
        df = pd.read_csv(title, sep = ';', dtype = np.float64)
        df = df[df["Particle ID"] != 0] # Delete formatting zeros

        # Get the name of the cluster from the file name
        clus_name = re.search("_([a-z]+)_Pulses.csv", title).group(1) 
        
        # If missing create a new directory to store the curves of the current cluster 
        if not os.path.exists(directory_to_populate + '/' + clus_name):
            os.makedirs(data_destination + '/curves/all/' + clus_name)   
            os.makedirs(data_destination + '/curves/train/' + clus_name)   
            os.makedirs(data_destination + '/curves/validation/' + clus_name)
        
        
        # Generation of the 5 curves for each particle of each cluster      
        for idx, group in df.groupby(["Particle ID"]):
            if count >= max_cc_instance_per_date: # Avoid to have too much data 
                break
            if idx != 0:
                for col in df.columns[1:]:
                    plt.figure(figsize=(7,3)) # Check format ok
                    plt.plot(group[col])
                    plt.axis('off')
                    plt.savefig(directory_to_populate + '/' + clus_name + '/' + date + '_' + col + '_' + str(count) + '.png', \
                                format='png', dpi=100)
                    plt.close()
                count += 1
                

        # Add the date of the extraction
        df["date"] = date
        df = df[['Particle ID', 'date']] # The other 5 columns have already been used to generate curves: no need to keep then
        df.set_index("Particle ID", inplace = True)
        pulse_data = pulse_data.append(df)
        
    clus_indices = set(pulse_data.index)
    len_pulse_data = len(pulse_data)
    del(pulse_data) # Free the memory to avoid Memory Errors
            
    # Extract the curves of noise particules
        
    title = [t for t in pulse_titles_default if re.search(date, t)][0] # Dirty
    df = pd.read_csv(title, sep = ';', dtype = np.float64)
    df.set_index("Particle ID", inplace = True)
    
    # Get the indices of the noise particles to generate the associated curves
    all_indices = df.index
    all_indices = set(all_indices)
    noise_indices = list(set(df.index) - set(clus_indices))
    noise = df.loc[noise_indices] # Keep only the noise particles 

    clus_name = "noise"
    
    # Generation of the 5 curves for each noise particle     
    for idx, group in df.groupby(["Particle ID"]):
        if count >= max_cc_instance_per_date: # Avoid to have too much data 
            break
        
        if idx != 0:
            for col in df.columns[1:]:
                plt.figure(figsize=(3,3)) # Store them as square images
                plt.plot(group[col])
                plt.axis('off')
                plt.savefig(directory_to_populate + '/' + clus_name + '/' + date + '_' + col + '_' + str(count) + '.png', \
                            format='png', dpi=100)
                plt.close()
            count += 1
            
        
        # Sanity check 
        if (len(noise) + len_pulse_data != len(df)):
            raise ValueError("It seems that some data have been lost during extraction")
            
        # Free useless memory
        del(df)
        del(noise)

            

