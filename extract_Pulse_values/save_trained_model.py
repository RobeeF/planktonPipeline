# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:32:19 2019

@author: Utilisateur
"""

import numpy as np
import os 
import pandas as pd
from keras.callbacks import ModelCheckpoint
import random


os.chdir('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline')
cluster_classes = pd.read_csv('nomenclature.csv')['Nomenclature'].tolist()
from keras_utils import ffnn_model, ffnn_model_w_len, model13

os.chdir('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from ffnn_functions import get_curves_values_data, scaler


raw_data_source = 'L1_FUMSECK'
cleaned_data_source = 'L2_FUMSECK'

###############################################################################################
################################ Final data saving ############################################
###############################################################################################

seed = 10133 # To have comparable datasets

###### Padded data
pad = True
X_pad, seq_len_list_pad, y_pad, pid_list_pad, le_pad  = get_curves_values_data(raw_data_source,\
                                        cleaned_data_source, cluster_classes,\
                                        pad = pad, seed = seed)

np.save('L3_FUMSECK/X_pad.npy', X_pad)
np.save('L3_FUMSECK/y_pad.npy', y_pad)
np.save('L3_FUMSECK/pid_list_pad.npy', pid_list_pad)
np.save('L3_FUMSECK/seq_len_list_pad.npy', seq_len_list_pad)

###### Interpolated
pad = False
X_interp, seq_len_list_interp, y_interp, pid_list_interp, le_interp  = get_curves_values_data(raw_data_source,\
                                        cleaned_data_source, cluster_classes, pad = pad, seed = seed)

X_interp_s = scaler(X_interp)

np.save('L3_FUMSECK/X_interp_scale.npy', X_interp_s)
np.save('L3_FUMSECK/X_interp.npy', X_interp)
np.save('L3_FUMSECK/y_interp.npy', y_interp)
np.save('L3_FUMSECK/pid_list_interp.npy', pid_list_interp)
np.save('L3_FUMSECK/seq_len_list_interp.npy', seq_len_list_interp)

# Save the mapping created by the encoder (impossible to re-feed it as a dict to the encoded while predicting...)
train_nomenc = pd.DataFrame({'Particle_class': le_pad.classes_, 'Particle_class_num': le_pad.transform(cluster_classes).astype(int)})
train_nomenc.to_csv('train_nomenclature.csv', index = False)


###############################################################################################
############################ Training and saving models #######################################
###############################################################################################

### Model 1: Classic ffnn with balanced, unscaled and padded data

X = np.load('L3_FUMSECK/X_pad.npy')
y = np.load('L3_FUMSECK/y_pad.npy')
pid_list = np.load('L3_FUMSECK/pid_list_pad.npy')
seq_len_list = np.load('L3_FUMSECK/seq_len_list_pad.npy')


# Fitting model on all labelled data
model = ffnn_model(X, y, dp = 0.2)

batch_size = 32
STEP_SIZE = (len(X) // batch_size) + 1 

w_check = ModelCheckpoint(filepath='tmp/weights_pad_local.hdf5', verbose = 1, save_best_only=True)
history = model.fit(X, y, steps_per_epoch = STEP_SIZE, epochs = 80, callbacks = [w_check])

# Saving model
model.save('FUMSECK_trained')


# Quick test
preds = np.argmax(model.predict(X), axis = 1)
true =  np.argmax(y, axis = 1)
pd.Series(preds).value_counts()
pd.Series(true).value_counts()

np.mean(true == preds)

#############################
### Model 1_bis: Classic ffnn with balanced, scaled and interpolated data
############################

X = np.load('L3_FUMSECK/X_interp_scale.npy')
y = np.load('L3_FUMSECK/y_interp.npy')
pid_list = np.load('L3_FUMSECK/pid_list_interp.npy')

# Split the data to avoid choosing an overfitted model
train_size = int(len(y) * 0.80) # Big training set in order to train properly
indices = random.sample(range(len(y)), train_size)
X_train, X_valid = X[indices], X[list(set(range(len(y))) - set(indices))] 
y_train, y_valid = y[indices], y[list(set(range(len(y))) - set(indices))]
pid_list_train, pid_list_valid = pid_list[indices], pid_list[list(set(range(len(y))) - set(indices))]

# Model fitting
model_1_bis = ffnn_model(X, y, dp = 0.2)

batch_size = 32
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

w_check = ModelCheckpoint(filepath='tmp/weights_pad_local.hdf5', verbose = 1, save_best_only=True)
history = model_1_bis.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = 80, callbacks = [w_check])

# Saving model
model_1_bis.load_weights("tmp/weights_pad_local.hdf5")
model_1_bis.save('FUMSECK_trained_scale')

###########################
### Model2: Classic ffnn with balanced, unscaled and padded data, that takes also the sequence length as input
##########################

model2 = ffnn_model_w_len(X, y, seq_len_list, dp = 0.2)

batch_size = 32
STEP_SIZE = (len(X) // batch_size) + 1 

history = model2.fit([X, seq_len_list[...,np.newaxis, np.newaxis]], y, steps_per_epoch = STEP_SIZE, epochs = 80)
model2.save('FUMSECK_trained_w_len')

###########################
### Model3: CNN + ffnn with balanced, scaled and interpolated data
##########################
X = np.load('L3_FUMSECK/X_interp_scale.npy')
y = np.load('L3_FUMSECK/y_interp.npy')
pid_list = np.load('L3_FUMSECK/pid_list_interp.npy')

# Split the data to avoid choosing an overfitted model
train_size = int(len(y) * 0.80) # Big training set in order to train properly
indices = random.sample(range(len(y)), train_size)
X_train, X_valid = X[indices], X[list(set(range(len(y))) - set(indices))] 
y_train, y_valid = y[indices], y[list(set(range(len(y))) - set(indices))]
pid_list_train, pid_list_valid = pid_list[indices], pid_list[list(set(range(len(y))) - set(indices))]

# Model fitting
cffnn = model13(X, y, dp = 0.2)

batch_size = 32
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

cffnn_check = ModelCheckpoint(filepath='tmp/weights_cffnn.hdf5', verbose = 1, save_best_only=True)
history = cffnn.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = 80, callbacks = [cffnn_check])

# Saving model
cffnn.load_weights("tmp/weights_cffnn.hdf5")
cffnn.save('cffnn_FUMSECK_trained_scale')


# Extract custom features
from keras.models import Model

cffnn.summary() # Manually get the name of the intermediate feature extraction layer
layer_name = 'global_average_pooling1d_1'
intermediate_layer_cffnn = Model(inputs=cffnn.input,
                                 outputs=cffnn.get_layer(layer_name).output)
extracted_features = intermediate_layer_cffnn.predict(X)

intermediate_layer_cffnn.save('Feature_Xor')
np.save('X_features', extracted_features)
