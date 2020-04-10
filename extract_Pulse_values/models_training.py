# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:32:19 2019

@author: Utilisateur
"""

import numpy as np
import os 
import pandas as pd

from time import time
from scipy.integrate import trapz
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline')
cluster_classes = pd.read_csv('nomenclature.csv')['Nomenclature'].tolist()

from keras_utils import  model13

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from dataset_prepocessing import gen_dataset, gen_train_test_valid
from viz_functions import plot_2D, plot_2Dcyto
from pred_functions import predict

raw_data_source = 'L1_FUMSECK'
cleaned_data_source = 'L2_FUMSECK'

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']


##############################################################################################
######################### Train Model 13 on FUMSECK Data ####################################
##############################################################################################

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours

source = "FUMSECK_L2_fp"
mnbf = 6 # Total number of files to extract
max_nb_files_extract = mnbf
nb_valid_files = 4
nb_test_files = 1
prop = [1 - (nb_valid_files + nb_test_files) / mnbf, nb_valid_files / mnbf, nb_test_files / mnbf]


start = time()
X_train, y_train, X_valid, y_valid, X_test, y_test = gen_train_test_valid(source, cluster_classes, mnbf, prop)
end = time()
print(end - start)


# Save the dataset (2 or 3 hours to extract it)
'''
np.save('FUMSECK_L3/X_train610', X_train)
np.save('FUMSECK_L3/y_train610', y_train)

np.save('FUMSECK_L3/X_valid610', X_valid)
np.save('FUMSECK_L3/y_valid610', y_valid)

np.save('FUMSECK_L3/X_test610', X_test)
np.save('FUMSECK_L3/y_test610', y_test)
'''

# Load data
X_train = np.load('FUMSECK_L3/X_train610.npy')
y_train = np.load('FUMSECK_L3/y_train610.npy')

X_valid = np.load('FUMSECK_L3/X_valid610.npy')
y_valid = np.load('FUMSECK_L3/y_valid610.npy')

X_test = np.load('FUMSECK_L3/X_test610.npy')
y_test = np.load('FUMSECK_L3/y_test610.npy')

#========================================
# (Optional) ENN : delete dirty examples
#========================================

X_integrated = trapz(X_train, axis = 1)
X_integrated = pd.DataFrame(X_integrated, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])
y = y_train.argmax(1)
  
# ENN for cleaning data
enn = EditedNearestNeighbours()
X_rs, y_rs = enn.fit_resample(X_integrated, y) 

X_train = X_train.take(enn.sample_indices_, axis = 0)
y_train = y_train.take(enn.sample_indices_, axis = 0)

#plot_2Dcyto(X_rs, y_rs, tn, 'FWS', 'FL Red')
#plot_2Dcyto(X_integrated, y, tn, 'FWS', 'FL Red')

#========================================================
# RUS: Delete random observations from majority classes
#========================================================

balancing_dict = Counter(np.argmax(y_train,axis = 1))
for class_, obs_nb in balancing_dict.items():
    if obs_nb > 3000:
        balancing_dict[class_] = 3000


rus = RandomUnderSampler(sampling_strategy = balancing_dict)
ids = np.arange(len(X_train)).reshape((-1, 1))
ids_rs, y_train = rus.fit_sample(ids, y_train)
X_train = X_train[ids_rs.flatten()] 


w = 1/ np.sum(y_valid, axis = 0)
#w[-2] = w[-2] * 1.2 # Make the weight of prochloro rise
w = np.where(w == np.inf, np.max(w[np.isfinite(w)]) * 2 , w)
w = w / w.sum() 


batch_size = 128
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

cffnn = model13(X_train, y_train, dp = 0.2)
ENN_check = ModelCheckpoint(filepath='tmp/weights_ENN.hdf5', verbose = 1, save_best_only=True)

epoch_nb = 1

history = cffnn.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check], class_weight = w, shuffle=True)

cffnn.load_weights('tmp/weights_ENN.hdf5')



#### Compute accuracies #####

# Compute train accuracy
preds = np.argmax(cffnn.predict(X_train), axis = 1)
true = np.argmax(y_train, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on train data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='weighted'))
print(confusion_matrix(true, preds))    

# Compute valid accuracy
preds = np.argmax(cffnn.predict(X_valid), axis = 1)
true = np.argmax(y_valid, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on valid data !', acc)
print('Weighted accuracy is', precision_score(true, preds, \
                        average='weighted', zero_division = 0))
print(confusion_matrix(true, preds))

# Compute test accuracy
start = time()
preds = np.argmax(cffnn.predict(X_test), axis = 1)
end = time()
print(end - start)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds))
#print('categorical ce is ', - np.sum(true * np.log(preds)))
    

# Good model : Save model
cffnn.save('ENN_LottyNet_FUMSECK')

############################################################################################################
################## Fine tune the LottyNet_FUMSECK on Endoume first week data ###############################
############################################################################################################

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from keras.models import load_model
import numpy as np
from keras.optimizers import adam
from keras import metrics

fumseck = load_model('trained_models/LottyNet_FUMSECK')

#=================================================
# Loading Endoume first week data (the dirty way)
#=================================================

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

#===============================================================
# Former model prediction for comparison
#===============================================================

predict(source + '/' + test_files[0], source, fumseck, tn)

preds = pd.read_csv(source + '/original/Pulse6_2019-09-18 15h59.csv')

q1 = 'Total FWS'
q2 = 'Total FLR'

plot_2D(preds, tn, q1, q2)

#===============================================================
# Model preparation for fine-tuning
#===============================================================

# Freeze the first layers and retrain
for layer in fumseck.layers[:5]:
    layer.trainable = False


ad = adam(lr=1e-3)
fumseck.compile(optimizer=ad, loss='categorical_crossentropy', \
                metrics=[metrics.categorical_accuracy])
    
#================================================================
# Data importation and model fitting
#================================================================

X_train_SLAAMM, seq_len_list_train_SLAAMM, y_train_SLAAMM, pid_list_train_SLAAMM, file_name_train_SLAAMM, le_train = gen_dataset(source, \
                            cluster_classes, train_files, None, nb_obs_to_extract_per_group = 100, \
                            pad = False, to_balance = True, seed = None)
    
X_valid_SLAAMM, seq_len_list_valid_SLAAMM, y_valid_SLAAMM, pid_list_valid, file_name_valid_SLAAMM, le_valid = gen_dataset(source, \
                            cluster_classes, valid_files, None, nb_obs_to_extract_per_group = 600, default_sampling_nb = 70,\
                            pad = False, to_balance = False, to_undersample = True, seed = None)

# Extract the test dataset from full files
X_test_SLAAMM, seq_len_list_test_SLAAMM, y_test_SLAAMM, pid_list_test_SLAAMM, file_name_test_SLAAMM, le_test = gen_dataset(source, \
                            cluster_classes, test_files, None, nb_obs_to_extract_per_group = 100, \
                            pad = False, to_balance = False, to_undersample = False, seed = None)

    
np.save('FUMSECK_L3/X_trainFLR6_SSLAMM', X_train_SLAAMM)
np.save('FUMSECK_L3/y_trainFLR6_SSLAMM', y_train_SLAAMM)

np.save('FUMSECK_L3/X_validFLR6_SSLAMM', X_valid_SLAAMM)
np.save('FUMSECK_L3/y_validFLR6_SSLAMM', y_valid_SLAAMM)

np.save('FUMSECK_L3/X_testFLR6_SSLAMM', X_test_SLAAMM)
np.save('FUMSECK_L3/y_testFLR6_SSLAMM', y_test_SLAAMM)


    
X_train_SLAAMM = np.load('FUMSECK_L3/X_trainFLR6_SSLAMM.npy')
y_train_SLAAMM = np.load('FUMSECK_L3/y_trainFLR6_SSLAMM.npy')

X_valid_SLAAMM = np.load('FUMSECK_L3/X_validFLR6_SSLAMM.npy')
y_valid_SLAAMM = np.load('FUMSECK_L3/y_validFLR6_SSLAMM.npy')

X_test_SLAAMM = np.load('FUMSECK_L3/X_testFLR6_SSLAMM.npy')
y_test_SLAAMM = np.load('FUMSECK_L3/y_testFLR6_SSLAMM.npy')
 

# Keep the weights w unchanged  

batch_size = 128
STEP_SIZE_TRAIN = (len(X_train_SLAAMM) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid_SLAAMM) // batch_size) + 1 

blcd_check = ModelCheckpoint(filepath='tmp/weights_fum_n_slaamm.hdf5', verbose = 1, save_best_only=True)

epoch_nb = 6
for i in range(epoch_nb):
    
    history = fumseck.fit(X_train_SLAAMM, y_train_SLAAMM, validation_data=(X_valid_SLAAMM, y_valid_SLAAMM), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = 1, callbacks = [blcd_check])


fumseck.load_weights("tmp/weights_fum_n_slaamm.hdf5")

#=================================
# Visualising the results
#==================================

predict(source + '/' + test_files[0], source, fumseck, tn)

preds = pd.read_csv(source + 're_train/Pulse6_2019-09-18 15h59.csv')

q1 = 'Total FWS'
q2 = 'Total FLR'

plot_2D(preds, tn, q1, q2)

