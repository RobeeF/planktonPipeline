# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:35 2019

@author: robin
"""

import os 
from collections import Counter

# The directory in which you have placed the following code
os.chdir('C:/Users/robin/Documents/GitHub/planktonPipeline')

from from_cytoclus_to_files import extract_features
from from_cytoclus_to_imgs import extract_imgs
from from_files_to_clusters import particle_clustering

# Where to look the data at and where to write treated data: Change with yours
data_source = 'C:/Users/robin/Documents/Documents_importants/scolarité/These/ROBIN'
data_destination = 'C:/Users/robin/Documents/Documents_importants/scolarité/These/pipeline/data'

# Extract the features and the data
extract_features(data_source, data_destination)
extract_imgs(data_source, data_destination)

# Make the clusters with k-means from the features and compare it to the true labels
files_titles = os.listdir(data_destination + '/features')
error_rates, preds_labels = particle_clustering(data_destination + '/features', clus_method = 'k-means')

for date in error_rates.keys():
    print("Sample extracted on ", date)
    print("K-MEANS error is ", error_rates[date])
    print("The following numbers of each particle types were found in the files")
    count = Counter(preds_labels[date]) 
    print(count)
    print('----------------------------------------------------')
    
for title in files_titles:
    df = pd.read_csv(data_destination + '/features/' + title)
    vc = df["cluster"].value_counts()
    vc = vc["noise"]/np.sum(vc) 
    print(vc)
