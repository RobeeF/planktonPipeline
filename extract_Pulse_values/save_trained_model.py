# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:32:19 2019

@author: Utilisateur
"""

import numpy as np
import os 
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Model

import random


os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline')
cluster_classes = pd.read_csv('nomenclature.csv')['Nomenclature'].tolist()
from keras_utils import ffnn_model, ffnn_model_w_len, model13, plot_losses

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from ffnn_functions import gen_dataset, scaler, gen_train_test_valid

raw_data_source = 'L1_FUMSECK'
cleaned_data_source = 'L2_FUMSECK'

###############################################################################################
################################ Final data saving ############################################
###############################################################################################

seed = 10133 # To have comparable datasets

###### Padded data
pad = True
X_pad, seq_len_list_pad, y_pad, pid_list_pad, le_pad  = gen_dataset(raw_data_source,\
                                        cleaned_data_source, cluster_classes,\
                                        pad = pad, seed = seed)

np.save('L3_FUMSECK/X_pad.npy', X_pad)
np.save('L3_FUMSECK/y_pad.npy', y_pad)
np.save('L3_FUMSECK/pid_list_pad.npy', pid_list_pad)
np.save('L3_FUMSECK/seq_len_list_pad.npy', seq_len_list_pad)

###### Interpolated
pad = False
X_interp, seq_len_list_interp, y_interp, pid_list_interp, le_interp  = gen_dataset(raw_data_source,\
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

cffnn.summary() # Manually get the name of the intermediate feature extraction layer
layer_name = 'global_average_pooling1d_1'
intermediate_layer_cffnn = Model(inputs=cffnn.input,
                                 outputs=cffnn.get_layer(layer_name).output)
extracted_features = intermediate_layer_cffnn.predict(X)

intermediate_layer_cffnn.save('Feature_Xor')
np.save('X_features', extracted_features)


##############################################################################################
# Fresh new start with all FUMSECK Data
##############################################################################################
from keras.losses import CategoricalCrossentropy as cc
from sklearn.metrics import confusion_matrix,classification_report
from time import time
from keras.utils.np_utils import to_categorical

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

source = "FUMSECK_L2_fp"
mnbf = 6
max_nb_files_extract = mnbf
nb_valid_files = 4
nb_test_files = 1
prop = [1 - (nb_valid_files + nb_test_files) / mnbf, nb_valid_files / mnbf, nb_test_files / mnbf]


start = time()
X_train, y_train, X_valid, y_valid, X_test, y_test = gen_train_test_valid(source, cluster_classes, mnbf, prop)
end = time()
print(end - start)


# Save the dataset (hours to extract it)
'''
np.save('FUMSECK_L3/X_train610', X_train)
np.save('FUMSECK_L3/y_train610', y_train)

np.save('FUMSECK_L3/X_valid610', X_valid)
np.save('FUMSECK_L3/y_valid610', y_valid)

np.save('FUMSECK_L3/X_test610', X_test)
np.save('FUMSECK_L3/y_test610', y_test)


# Load data
X_train = np.load('FUMSECK_L3/X_train610.npy')
y_train = np.load('FUMSECK_L3/y_train610.npy')

X_valid = np.load('FUMSECK_L3/X_valid610.npy')
y_valid = np.load('FUMSECK_L3/y_valid610.npy')

X_test = np.load('FUMSECK_L3/X_test610.npy')
y_test = np.load('FUMSECK_L3/y_test610.npy')
'''

# If want to scale
#X_train = scaler(X_train)
#X_test = scaler(X_train)
#X_valid = scaler(X_valid)

# Forced to undersample: 300 files is too much, even for the light version of the model
# To add directly in gen_train_test_valid 
from collections import Counter
balancing_dict = Counter(np.argmax(y_train,axis = 1))
for class_, obs_nb in balancing_dict.items():
    if obs_nb > 3000:
        balancing_dict[class_] = 3000

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(ratio = balancing_dict)
            
ids = np.arange(len(X_train)).reshape((-1, 1))
ids_rs, y_train = rus.fit_sample(ids, np.argmax(y_train, axis = 1))
X_train = X_train[ids_rs.flatten()] 

y_train =  to_categorical(y_train, num_classes = len(cluster_classes))

w = 1/ np.sum(y_valid, axis = 0)
w = np.where(w == np.inf, np.max(w) * 2 , w)
w = w/ w.sum() # Not enough weight on picoeucaryote #4

batch_size = 128
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

cffnn = model13(X_train, y_train, dp = 0.2)
blcd_check = ModelCheckpoint(filepath='tmp/weights_blcd.hdf5', verbose = 1, save_best_only=True)

cffnn.load_weights("tmp/weights_blcd.hdf5")

epoch_nb = 2
for i in range(epoch_nb):
    
    history = cffnn.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = 1, callbacks = [blcd_check], class_weight = w)


#### Compute accuracies #####

# Compute train accuracy
preds = np.argmax(cffnn.predict(X_train), axis = 1)
true = np.argmax(y_train, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on train data !', acc)
print(confusion_matrix(true, preds))    

# Compute valid accuracy
preds = np.argmax(cffnn.predict(X_valid), axis = 1)
true = np.argmax(y_valid, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on valid data !', acc)
print(confusion_matrix(true, preds))


# Compute test accuracy
start = time()
preds = np.argmax(cffnn.predict(X_test), axis = 1)
end = time()
print(end - start)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print(confusion_matrix(true, preds))
#print('categorical ce is ', - np.sum(true * np.log(preds)))
    

# Good model : Save model
cffnn.save('LottyNet_FUMSECK')
len(y_test)


####### Test for confusion matrix
import matplotlib.pyplot as plt
lab_tab = pd.read_csv(source + '/train_test_nomenclature.csv').set_index('labels')['cluster'].to_dict()
labels = cluster_classes
cm = confusion_matrix([lab_tab[x] for x in true], [lab_tab[x] for x in preds], cluster_classes)
cm = cm/cm.sum(axis = 1, keepdims = True)
cm = np.where(np.isnan(cm), 0, cm)
print(cm) 

fig = plt.figure(figsize = (14,14)) 
ax = fig.add_subplot(111) 
cax = ax.matshow(cm) 
plt.title('Confusion matrix of LottyNet_Full on a FLR6 file') 
fig.colorbar(cax) 
ax.set_xticklabels([''] + labels) 
ax.set_yticklabels([''] + labels) 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.show()