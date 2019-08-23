# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:22:26 2019

@author: RFuchs
"""

import os 
import re
import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

def get_imgs_data(listmodes_path, imgs_dirs_path, flr_num = 25):
    # Get the id/cluster type     
    files_title = [f for f in os.listdir(listmodes_path)]
    flr_title = [f for f in files_title if re.search("^FUMSECK-FLR" + str(flr_num),f) and re.search("csv",f) ]
    
    listmode_titles_clus = [f for f in flr_title if  re.search("Listmode",f) and not(re.search("Default",f))] # Gerer les default

    listmode_regex = "_([a-zA-Z0-9 ]+)_Listmode.csv"
    date_regex = "FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"
        
    X = []
    y = [] 
    pid_list = []
    
    # Build the id/cluster type table        
    for repo in os.listdir(imgs_dirs_path):
        print(repo)
        date = re.search(date_regex, repo).group(1)
        date_datasets_titles = [t for t in listmode_titles_clus if re.search(date, t)]

        ids_type_df = pd.DataFrame()
        for title in date_datasets_titles:
            try:
                df = pd.read_csv(listmodes_path + '/' + title, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                df = pd.read_csv(listmodes_path + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',') 
            
            clus_name = re.search(listmode_regex, title).group(1) 
            df['cluster'] = clus_name
            df['Particle ID'] = df['Particle ID'].astype(int)
            ids_type_df = ids_type_df.append(df[['Particle ID', 'cluster']])
        
        
        ids_type_df.set_index('Particle ID', inplace = True)
        
        # Fetch the labels to the images
        for img_title in os.listdir(imgs_dirs_path + '/' + repo):
            if re.search("Uncropped", img_title):
                continue
            
            img = imread(imgs_dirs_path + '/' + repo + '/' + img_title)
            img = rgb2gray(img)
            img = resize(img, (90, 130), mode='reflect', preserve_range=True)
            X.append(img)
            
            pid = re.search('([0-9]+).tif', img_title).group(1) 
            try:
                y.append(ids_type_df.loc[float(pid)]['cluster'])
            except:
                y.append('noise')
            pid_list.append(pid)
        
    X = np.stack(X)
    
    # Encode the names of the clusters into integers. Does it here to be consistant with the other scripts (Pulse values, curves etc...)
    le = LabelEncoder()
    print(list(set(y)))
    le.fit(list(set(y))) 
    
    y = le.transform(y)
    y =  to_categorical(y)
    
    return X, y, pid_list, le