# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:11:29 2020

@author: rfuchs
"""

import os
import numpy as np
import pandas as pd


os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from time import time
from collections import Counter
from dataset_preprocessing import gen_dataset, gen_train_test_valid


os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline')
cluster_classes = pd.read_csv('nomenclature.csv')['Nomenclature'].tolist()

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

##############################################################################################
################################# Full FUMSECK Data ##########################################
##############################################################################################
source = "FUMSECK_L2_fp"
mnbf = 6 # Total number of files to extract
max_nb_files_extract = mnbf
nb_valid_files = 4
nb_test_files = 1
prop = [1 - (nb_valid_files + nb_test_files) / mnbf, nb_valid_files / mnbf, nb_test_files / mnbf]


start = time()
X_train, y_train, X_valid, y_valid, X_test, y_test = gen_train_test_valid(source, cluster_classes, mnbf, prop)
end = time()
print(end - start) # About 2 or 3 hours to extract it


# Save the dataset 

np.save('FUMSECK_L3/X_train610', X_train)
np.save('FUMSECK_L3/y_train610', y_train)

np.save('FUMSECK_L3/X_valid610', X_valid)
np.save('FUMSECK_L3/y_valid610', y_valid)

np.save('FUMSECK_L3/X_test610', X_test)
np.save('FUMSECK_L3/y_test610', y_test)

############################################################################################################
################################### FLR6 from SSLAMM first week ############################################
############################################################################################################

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

source = "SSLAMM/Week1"
mnbf = 6 # Total number of files to extract
max_nb_files_extract = mnbf
nb_valid_files = 4
nb_test_files = 1
prop = [1 - (nb_valid_files + nb_test_files) / mnbf, nb_valid_files / mnbf, nb_test_files / mnbf]


files = os.listdir(source)
train_files = files[2:]
valid_files = [files[0]]
test_files = [files[1]] # Last one one the 21 files


X_train_SLAAMM, seq_len_list_train_SLAAMM, y_train_SLAAMM, pid_list_train_SLAAMM, file_name_train_SLAAMM, le_train = gen_dataset(source, \
                            cluster_classes, train_files, None, nb_obs_to_extract_per_group = 100, \
                            to_balance = True, seed = None)
    
X_valid_SLAAMM, seq_len_list_valid_SLAAMM, y_valid_SLAAMM, pid_list_valid, file_name_valid_SLAAMM, le_valid = gen_dataset(source, \
                            cluster_classes, valid_files, None, nb_obs_to_extract_per_group = 600, default_sampling_nb = 70,\
                            to_balance = False, to_undersample = True, seed = None)

# Extract the test dataset from full files
X_test_SLAAMM, seq_len_list_test_SLAAMM, y_test_SLAAMM, pid_list_test_SLAAMM, file_name_test_SLAAMM, le_test = gen_dataset(source, \
                            cluster_classes, test_files, None, nb_obs_to_extract_per_group = 100, \
                            to_balance = False, to_undersample = False, seed = None)

    
np.save('FUMSECK_L3/X_trainFLR6_SSLAMM', X_train_SLAAMM)
np.save('FUMSECK_L3/y_trainFLR6_SSLAMM', y_train_SLAAMM)

np.save('FUMSECK_L3/X_validFLR6_SSLAMM', X_valid_SLAAMM)
np.save('FUMSECK_L3/y_validFLR6_SSLAMM', y_valid_SLAAMM)

np.save('FUMSECK_L3/X_testFLR6_SSLAMM', X_test_SLAAMM)
np.save('FUMSECK_L3/y_testFLR6_SSLAMM', y_test_SLAAMM)


##############################################################################################
################################## Full SSLAMM data ##########################################
##############################################################################################

cluster_classes = ['airbubble', 'cryptophyte', 'hsnano', 'microphytoplancton',
       'nanoeucaryote', 'picoeucaryote', 'prochlorococcus',
       'synechococcus', 'unassigned particle']


source = "SSLAMM_L2"
nb_train_files = 24
nb_valid_files = 4
nb_test_files = 4

nb_files_tvt = [nb_train_files, nb_valid_files, nb_test_files]

start = time()
X_train, y_train, X_valid, y_valid, X_test, y_test = gen_train_test_valid(source, cluster_classes, nb_files_tvt, \
                                                                          train_umbal_margin = 7000, seed = None)
end = time()
print(end - start) # About 8 minutes

len(Counter(y_train.argmax(1)))
len(Counter(y_valid.argmax(1)))
len(Counter(y_test.argmax(1)))


np.save('SSLAMM_L3/X_train', X_train)
np.save('SSLAMM_L3/y_train', y_train)

np.save('SSLAMM_L3/X_valid', X_valid)
np.save('SSLAMM_L3/y_valid', y_valid)

np.save('SSLAMM_L3/X_test', X_test)
np.save('SSLAMM_L3/y_test', y_test)


##############################################################################################
###################### SSLAMM few files imbalanced for focal #################################
##############################################################################################

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

source = "SSLAMM_L2"
source = "SSLAMM_L2"
nb_train_files = 24
nb_valid_files = 4
nb_test_files = 4

nb_files_tvt = [nb_train_files, nb_valid_files, nb_test_files]

files = os.listdir(source)

train_files = files[0:3]
train_files = train_files + [files[-2]]

valid_files = files[3:6]
valid_files = valid_files + [files[-3]]

test_files = [files[-1]] 

cluster_classes = ['airbubble', 'cryptophyte', 'hsnano', 'microphytoplancton',
       'nanoeucaryote', 'picoeucaryote', 'prochlorococcus',
       'synechococcus', 'unassigned particle']

X_train, seq_len_list_train, y_train, pid_list_train, file_name_train, le_train = gen_dataset(source, \
                            cluster_classes, train_files, None, nb_obs_to_extract_per_group = 600, default_sampling_nb = 70,\
                            to_balance = False, to_undersample = True, seed = None)
    
X_valid, seq_len_list_valid, y_valid, pid_list_valid, file_name_valid, le_valid = gen_dataset(source, \
                            cluster_classes, valid_files, None, nb_obs_to_extract_per_group = 600, default_sampling_nb = 70,\
                            to_balance = False, to_undersample = True, seed = None)

# Extract the test dataset from full files
X_test, seq_len_list_test, y_test, pid_list_test, file_name_test, le_test = gen_dataset(source, \
                            cluster_classes, test_files, None, nb_obs_to_extract_per_group = 100, \
                            to_balance = False, to_undersample = False, seed = None)
    
    
    
np.save('SSLAMM_L3/X_train_umbal_SLAMM', X_train)
np.save('SSLAMM_L3/y_train_umbal_SLAMM', y_train)

np.save('SSLAMM_L3/X_valid_umbal_SLAMM', X_valid)
np.save('SSLAMM_L3/y_valid_umbal_SLAMM', y_valid)

np.save('SSLAMM_L3/X_test_umbal_SLAMM', X_test)
np.save('SSLAMM_L3/y_test_umbal_SLAMM', y_test)
    
##############################################################################################
######### Enriched few file data SSLAMM data with airbubbles and microphyto ##################
##############################################################################################


# Import and keep Airbubbles and microphyto from FUMSECK
## Load data
X_train_F = np.load('FUMSECK_L3/X_train610.npy')
y_train_F = np.load('FUMSECK_L3/y_train610.npy')

X_valid_F = np.load('FUMSECK_L3/X_valid610.npy')
y_valid_F = np.load('FUMSECK_L3/y_valid610.npy')

X_test_F = np.load('FUMSECK_L3/X_test610.npy')
y_test_F = np.load('FUMSECK_L3/y_test610.npy')

np.random.seed = 0

## Select from the 3 sets 
proc_micro_idx = np.logical_or((y_train_F.argmax(1) == 0), (y_train_F.argmax(1) == 2))
X_micro_obs = X_train_F[proc_micro_idx]
y_micro_obs = y_train_F[proc_micro_idx]

proc_micro_idx = np.logical_or((y_valid_F.argmax(1) == 0), (y_valid_F.argmax(1) == 2))
X_micro_obs = np.concatenate([X_micro_obs, X_valid_F[proc_micro_idx]], axis = 0)
y_micro_obs = np.concatenate([y_micro_obs, y_valid_F[proc_micro_idx]], axis = 0)


proc_micro_idx = np.logical_or((y_test_F.argmax(1) == 0), (y_test_F.argmax(1) == 2))
X_micro_obs = np.concatenate([X_micro_obs, X_test_F[proc_micro_idx]], axis = 0)
y_micro_obs = np.concatenate([y_micro_obs, y_test_F[proc_micro_idx]], axis = 0)

# Nomenclature change, shift the microphyto column by one 
y_micro_obs = np.insert(y_micro_obs, 2, values=0, axis=1)

idx = list(range(len(y_micro_obs)))
np.random.shuffle(idx)


# Load training set from SSLAMM
X_train_SS = np.load('SSLAMM_L3/X_train_umbal_SLAMM.npy')
y_train_SS = np.load('SSLAMM_L3/y_train_umbal_SLAMM.npy')

X_valid_SS = np.load('SSLAMM_L3/X_valid_umbal_SLAMM.npy')
y_valid_SS = np.load('SSLAMM_L3/y_valid_umbal_SLAMM.npy')

X_test_SS = np.load('SSLAMM_L3/X_test_umbal_SLAMM.npy')
y_test_SS = np.load('SSLAMM_L3/y_test_umbal_SLAMM.npy')


# Enrich training set of SSLAMM with the FUMSECK data
X_hybrid_train = np.concatenate([X_train_SS, X_micro_obs[idx[ :2000]]], axis = 0)
y_hybrid_train = np.concatenate([y_train_SS, y_micro_obs[idx[ :2000]]], axis = 0)

X_hybrid_valid = np.concatenate([X_valid_SS, X_micro_obs[idx[2000: 3010]]], axis = 0)
y_hybrid_valid = np.concatenate([y_valid_SS, y_micro_obs[idx[2000: 3010]]], axis = 0)

X_hybrid_test = np.concatenate([X_test_SS, X_micro_obs[idx[3010: ]]], axis = 0)
y_hybrid_test = np.concatenate([y_test_SS, y_micro_obs[idx[3010: ]]], axis = 0)


# Save the hybrid dataset 
np.save('hybrid_L3/X_train', X_hybrid_train)
np.save('hybrid_L3/y_train', y_hybrid_train)

np.save('hybrid_L3/X_valid', X_hybrid_valid)
np.save('hybrid_L3/y_valid', y_hybrid_valid)

np.save('hybrid_L3/X_test', X_hybrid_test)
np.save('hybrid_L3/y_test', y_hybrid_test)


#########################################################################################
############### Count the number of Particle of each type in the dataset ################
#########################################################################################

import re
import fastparquet as fp
os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

def get_first_el(list_):
    return list_[0]


source = "SSLAMM_L2"

files = os.listdir(source)
files = [f for f in files if re.search("Labelled",f) and not(re.search('lock', f))]

hsnano = 0
nano = 0
airbubbles = 0
picoeucaryotes = 0
Prochlorococcus = 0
synechococcus = 0
una = 0
microphyto = 0
crypto = 0
tot_nb_part = 0

for file in files:
    pfile = fp.ParquetFile(source + '/' + file)
    df = pfile.to_pandas(columns = ['cluster', 'Particle ID'])
    df = df.groupby('Particle ID').apply(np.unique).apply(get_first_el)
    tot_nb_part += len(df)
    hsnano += np.sum(df == 'HSnano')
    nano += np.sum(df == 'nanoeucaryotes')
    airbubbles += np.sum(df == 'airbubbles')
    picoeucaryotes += np.sum(df == 'picoeucaryotes')
    Prochlorococcus += np.sum(df == 'Prochlorococcus')
    synechococcus += np.sum(df == 'synechococcus')
    una += np.sum(df == 'Unassigned Particles')
    crypto += np.sum(df == 'cryptophytes')
    microphyto += np.sum(df == 'microphytoplancton')
    
    
crypto / 32