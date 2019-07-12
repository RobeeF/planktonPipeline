# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:04:34 2019

@author: robin
"""

import pandas as pd
import os 
import re

#============================================================================
# Listmode features extraction
#============================================================================

def extract_features(data_source, data_destination, flr_num = 6):
    ''' Create a csv file for every sample containing the features and the label of each particle 
    data_source (str): The path to the original source of data
    data_features (str) : The path to write the data to
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    ---------------------------------------------------------------------------------------------
    returns (None): Write the files on hard disk
    '''
    
    assert (flr_num == 6) or (flr_num == 25)    
    
    # Keep only the csv files of interest
    files_title = [f for f in os.listdir(data_source)]
    flr_title = [f for f in files_title if re.search("^FUMSECK-FLR" + str(flr_num),f) and re.search("csv",f) ]
    print(flr_title)

    listmode_titles_clus = [f for f in flr_title if  re.search("Listmode",f) and not(re.search("Default",f))] # Gerer les default
    listmode_titles_default = [f for f in flr_title if  re.search("Listmode",f) and re.search("Default",f)] # Gerer les default

    # Extract the timestamps of extractions and the labelled of identified clusters
    dates = set([re.search("FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z]+" , f).group(1) for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)])
    cluster_classes = list(set([re.search("_([a-z]+)_Listmode.csv", cc).group(1) for cc in listmode_titles_clus]))
    cluster_classes += ['noise']
    
    # create the directories that will host the curves
    if not os.path.exists(data_destination + '/features'):
        os.makedirs(data_destination + '/features') 
    
    ### From the original datafiles create a new csv per date: date, particle id, particle features, cluster id  
    for date in dates: 
        listmode_data = pd.DataFrame()
    
        ## Get the info about each "actual" particule and its cluster label
        date_datasets_titles = [t for t in listmode_titles_clus if re.search(date, t)]
        for title in date_datasets_titles:        
            df = pd.read_csv(data_source + '/' + title, sep = ';')
            # Add the name of the cluster from the file name
            clus_name = re.search("_([a-z]+)_Listmode.csv", title).group(1) 
            df["cluster"] = clus_name
            
            # Add the date of the extraction
            date = re.search("FLR6 (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[a-z]+", title).group(1) 
            df["date"] = date
        
            df.set_index("Particle ID", inplace = True)
            listmode_data = listmode_data.append(df)
            
        ## Extract info of noise particles 
        title = [t for t in listmode_titles_default if re.search(date, t)][0] # Dirty
        df = pd.read_csv(data_source + '/' + title, sep = ';')
        df["date"] = date
    
        date = re.search("FLR6 (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z]+", title).group(1) 
        
        existing_indices = listmode_data[listmode_data['date']==date].index
        
        df.set_index("Particle ID", inplace = True)
        noise_indices = list(set(df.index) - set(existing_indices)) # Determining the noise particles indices
        df = df.iloc[noise_indices] # Keep only the noise particles 
    
        clus_name = "noise"
        df["cluster"] = clus_name
    
        listmode_data = listmode_data.append(df)
        
        listmode_data.to_csv(data_destination + '/features/Labelled_' + str(flr_num) + '_' + date + '.csv') # Store the data on hard disk
        
