# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:35 2019

@author: robin
"""

import os 
from collections import Counter
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import plot_model

# The directory in which you have placed the following code
os.chdir('W:/Bureau/these/planktonPipeline')

from from_cytoclus_to_files import extract_features
from from_cytoclus_to_imgs import extract_imgs
from from_files_to_clusters import particle_clustering
from from_imgs_to_keras import imgs_train_test_valid_split
from img_recognition import toy_model, multi_input_data_generator

# Where to look the data at and where to write treated data: Change with yours
data_source = 'W:/Bureau/these/donnees_oceano'
data_destination = 'W:/Bureau/these/data'

#========== Make the clusters with k-means from the features and compare it to the true labels ============================#
# Extract the features
extract_features(data_source, data_destination)


files_titles = os.listdir(data_destination + '/features')
df = pd.read_csv(data_destination + '/features/' + files_titles[0], sep = ';', engine = 'python')
print(df.columns)


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
    
#========================== Train a supervised Convolutional Neural Network on the curves ================================#
pcles_per_date_to_extract = 50
# Extract the curves
extract_imgs(data_source, data_destination, pcles_per_date_to_extract)

# Split the curves in three sub-directories
root_dir = "W:/Bureau/these/data/curves"
imgs_train_test_valid_split(root_dir) 

# Train the network
model = toy_model()
#model.summary()
#plot_model(model, to_file='toy_model.png')

# For the moment the generator only output existing images. But rotated one etc can be generated
source_generator = ImageDataGenerator(horizontal_flip = True) # Allow 180° rotations
batch_size = 32

train_generator = multi_input_data_generator(source_generator, 'train', batch_size)
test_generator = multi_input_data_generator(source_generator, 'test', batch_size)
valid_generator = multi_input_data_generator(source_generator, 'valid', batch_size)

# 900 observations
STEP_SIZE_TRAIN = (46 * 7) // batch_size
STEP_SIZE_VALID = (46 * 7) // batch_size

os.chdir(root_dir)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs = 20
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


