# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:35 2019

@author: robin
"""

import os 
from collections import Counter
import pandas as pd
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE


#from xgboost import XGBClassifier


#from keras.utils import plot_model

# The directory in which you have placed the following code
os.chdir('W:/Bureau/these/planktonPipeline')

from from_cytoclus_to_files import extract_features
from from_cytoclus_to_imgs import extract_imgs
from from_files_to_clusters import particle_clustering
from from_imgs_to_keras import imgs_train_test_valid_split, nb_available_imgs
from img_recognition import toy_model, plot_losses, multi_input_gen, fit_gen



# Where to look the data at and where to write treated data: Change with yours
data_source = 'W:/Bureau/these/donnees_oceano'
data_destination = 'W:/Bureau/these/data'

seed = 42

#========== Make the clusters with k-means from the features and compare it to the true labels ============================#
# Extract the features
extract_features(data_source, data_destination, flr_num = 6)


files_titles = os.listdir(data_destination + '/features')
df = pd.read_csv(data_destination + '/features/' + files_titles[0], sep = ',', engine = 'python')
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
# Extract the curves
extract_imgs(data_source, data_destination, max_cc_instance_per_date = 2000, noise_pcles_to_load =  2 * (10 ** 6))

# Split the curves in three sub-directories
root_dir = "W:/Bureau/these/data/curves"
imgs_train_test_valid_split(root_dir) 

# Train the network
model = toy_model()
#model.summary()
#plot_model(model, to_file='toy_model.png')

# For the moment the generator only output existing images. But rotated one etc can be generated
source_generator = ImageDataGenerator(horizontal_flip = True) # Allow 180° rotations
batch_size = 512
os.chdir(root_dir)

train_generator_dict = multi_input_gen(source_generator, 'train', batch_size, shuffle = False)
test_generator_dict = multi_input_gen(source_generator, 'test', batch_size, shuffle = False)
valid_generator_dict = multi_input_gen(source_generator, 'valid', batch_size, shuffle = False)

train_generator = fit_gen(train_generator_dict)
test_generator = fit_gen(test_generator_dict)
valid_generator = fit_gen(valid_generator_dict)

#gens_dict = train_generator_dict

# Defining the number of steps in an epoch
nb_train, nb_test, nb_valid = nb_available_imgs(root_dir) # Compute the available images for each train, test and valid folder

STEP_SIZE_TRAIN = (nb_train // batch_size) + 1
STEP_SIZE_VALID = (nb_test // batch_size) + 1
STEP_SIZE_TEST = (nb_valid // batch_size) + 1 

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs = 5
)

import keras.backend as K
model.summary()
print(K.eval(model.optimizer.lr))

# General picture of the valid and train losses through the epochs:
plot_losses(history)

# What is the performance of the model for each class ? 
### Fetch the true labels of the test set
y_true = list(test_generator_dict.values())[0].classes
y_true.shape
## Get the predicted labels from the model
preds = model.predict_generator(test_generator, steps = STEP_SIZE_TEST)
y_pred = np.array([np.argmax(x) for x in preds])

np.mean(y_pred)

## Assess where does the model performs well and where it is not
print(classification_report(y_true, y_pred))
y_pred.shape


#============================== Train a supervised XGBoost on the features ===========================================#

files_titles = os.listdir(data_destination + '/features')
train = pd.read_csv(data_destination + '/features/' + files_titles[0], sep = ',', engine = 'python')
print(df.columns)

train.set_index(['Particle ID', 'date'], inplace = True)
train = train.dropna(how = 'any')

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
        
# Label Encoding: Turns the labels into numbers
le = LabelEncoder()
le.fit(list(set(Y)))
y = le.transform(Y)


sm = SMOTE(random_state = seed)
X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

#X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=seed)

clf = RandomForestClassifier(n_estimators=100, max_depth = 2,
                                 random_state=0, n_jobs = -1)
#clf.fit(X_train, y_train)  

clf.fit(X_res, y_res)

# On a second set
test = pd.read_csv(data_destination + '/features/' + files_titles[1], sep = ',', engine = 'python')
print(test.columns)

test.set_index(['Particle ID', 'date'], inplace = True)
test = test.dropna(how = 'any')

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Label Encoding: Turns the labels into numbers
le = LabelEncoder()
le.fit(list(set(y_test)))
y_test = le.transform(y_test)

y_pred = clf.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))

# On a third set
test = pd.read_csv(data_destination + '/features/' + files_titles[2], sep = ',', engine = 'python')
print(test.columns)

test.set_index(['Particle ID', 'date'], inplace = True)
test = test.dropna(how = 'any')

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Label Encoding: Turns the labels into numbers
le = LabelEncoder()
le.fit(list(set(y_test)))
y_test = le.transform(y_test)


y_pred = clf.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred, target_names=le.classes_))
help(classification_report)



