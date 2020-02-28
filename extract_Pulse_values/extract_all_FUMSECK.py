# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""
import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from from_cytoclus_to_curves_values import extract_curves_values
from ffnn_functions import gen_balanced_train
import pandas as pd
import numpy as np

# Extract the FLR 6
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 6
extract_curves_values(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


# Extract the FLR 25
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 25
extract_curves_values(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


# Create a balanced dataset
os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

clean_source = data_destination
cluster_classes = pd.read_csv('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/nomenclature.csv')['Nomenclature'].to_list()
X, seq_len_list, y, pid_list, file_name, le = gen_balanced_train(clean_source, cluster_classes)

np.save('FUMSECK_L3/X_toy', X)
np.save('FUMSECK_L3/y_toy', y)
np.save('FUMSECK_L3/seq_len_list_toy', seq_len_list)
np.save('FUMSECK_L3/pid_list_toy', pid_list)
np.save('FUMSECK_L3/file_name_toy', file_name)