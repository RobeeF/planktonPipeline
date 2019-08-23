# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:03:04 2019

@author: robin
"""

from keras.layers import Input, Dense, Conv1D, Concatenate, GlobalAveragePooling1D, Dropout, MaxPooling1D, LSTM, Flatten
from keras.models import Model
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np


#=================================================================================================================#
# CNN Utils 
#=================================================================================================================#


def moving_average_batch(a, n=3, pad_value = 1.):
    ''' Compute the moving average on every line of a 2D array and keep the same shape
    a (2d-array): The array to transform
    n (int): The size of the window
    pad_value (float): The value used to pad the moving averaged sequences
    ------------------------------------------------------------
    returns (2d-array): The transformed array
    '''
    assert len(a.shape) == 3
    batch_size = a.shape[0]
    height = a.shape[1]
    
    ret = np.cumsum(a, dtype=float, axis = 2)
    ret[:, :, n:] = ret[:, : ,n:] - ret[:, :, :-n]
    ma = ret[:, :, n - 1:] / n
    
    pad = np.full((batch_size, height, (n - 1) // 2), pad_value) 
    padded_ma = np.concatenate([pad, ma, pad], axis = 2)
    if padded_ma.shape != a.shape: # Happens when (width - n + 1) is an odd number
        padded_ma = np.concatenate([padded_ma, np.full((batch_size, height, 1), pad_value)], axis = 2)

    return padded_ma.astype('float32')

def cnn_model(nb_classes, img_height = 200, img_width = 300):
    ''' First simple Keras model to test the approach
    img_height (int): Height of the 5 curves images given as inputs
    img_width (int): Width of the 5 curves images given as inputs    
    ------------------------------------------------------------------------
    returns (Keras model): The callabel Toy Model
    '''
    # Inputs: 5 curves
    img_1 = Input(shape=(img_height, img_width), dtype='float32')
    img_2 = Input(shape=(img_height, img_width), dtype='float32')
    img_3 = Input(shape=(img_height, img_width), dtype='float32')
    img_4 = Input(shape=(img_height, img_width), dtype='float32')
    img_5 = Input(shape=(img_height, img_width), dtype='float32')
    
    # A level of 5 convolutional layers
    conv_1_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_1)  
    conv_2_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_2)  
    conv_3_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_3)  
    conv_4_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_4)  
    conv_5_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_5)  
        
    # Concatenate: The parameters deriving from the 5 curves of concatenated
    concat = Concatenate()([conv_1_1, conv_2_1, conv_3_1, conv_4_1, conv_5_1])
    
    dense1 = Dense(64, activation='relu')(concat)
    avgpool1 = GlobalAveragePooling1D()(dense1)
    predictions = Dense(nb_classes, activation='softmax')(avgpool1)
    
    model = Model(inputs=[img_1, img_2, img_3, img_4, img_5], outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def multi_input_gen(source_generator, repo, batch_size, shuffle = False, img_height = 200, img_width = 300):
        
    gens_dict = {} # The generator for each curve type will be holden in a dict
    curves_types = ['Curvature','FWS','SWS','FL Orange','FL Red']
    for curve_type in curves_types: 
        gen = source_generator.flow_from_directory(
            directory= './' + repo + '/' + curve_type + '/',
            target_size=(img_height, img_width),
            color_mode="grayscale",
            batch_size = batch_size,
            class_mode="categorical",
            shuffle = shuffle,
            seed=42
        )
        gens_dict[curve_type] = gen
    return gens_dict
   
     
def fit_gen(gens_dict, add_dim = False, ma = 1):
    # Batches of images are successively sent to the network
    img_batches = {}    
    while True:
        for curve_type in gens_dict.keys():         
            img_batches[curve_type] = gens_dict[curve_type].next()
        imgs_labels = list(img_batches.values()) # A bunch of batch_size * 5 curves and their label
        #imgs = [moving_average_batch(preprocess_input(imgs_labels[i][0][:,:,:,0]), ma) for i in range(len(imgs_labels))] # Extracting the curve images.
        imgs = [moving_average_batch(imgs_labels[i][0][:,:,:,0], ma) for i in range(len(imgs_labels))] # Extracting the curve images.


        if add_dim == True:
            imgs = [imgs]
        labels = [imgs_labels[i][1] for i in range(len(imgs_labels))] # Extracting the label corresponding to the set of 5 curves
        assert all([np.mean(label == labels[0]) == 1.0 for label in labels]) # Checking that the 5 curves extracted belong to the same group for each observation
        yield imgs, labels[0] # The label is the same for the 5 curves, so return only the label of the first curve
        

#=================================================================================================================#
# FFNN Utils 
#=================================================================================================================#

def ffnn_model(X, y, dp = 0.2):
    ''' Create a Feed Forward Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    average = GlobalAveragePooling1D()(sequence_input)
    dense1 = Dense(64, activation='relu')(average)
    drop1 = Dropout(dp)(dense1)
    dense2 = Dense(32, activation='relu')(drop1)
    drop2 = Dropout(dp)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(dp)(dense3)
    dense4 = Dense(16, activation='relu')(drop3)
    drop4 = Dropout(dp)(dense4)

    predictions = Dense(N_CLASSES, activation='softmax')(drop4)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model

def lstm_model(X, y):
    ''' Create a LSTM and Convolutional layers based model from O. Grisel Lecture-labs notebook
    X (ndarray): The features
    y (ndarray): The labels 
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    # input: a sequence of MAX_SEQUENCE_LENGTH integers
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # 1D convolution with 64 output channels
    x = Conv1D(64, 5)(sequence_input)
    
    # MaxPool divides the length of the sequence by 5: this is helpful
    # to train the LSTM layer on shorter sequences. The LSTM layer
    # can be very expensive to train on longer sequences.
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5)(x)
    x = MaxPooling1D(5)(x)
    
    # LSTM layer with a hidden size of 64
    x = LSTM(64)(x)

    predictions = Dense(N_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model

def conv_model(X, y):
    ''' Create a Convolutional layers based model
    X (ndarray): The features
    y (ndarray): The labels 
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    # input: a sequence of MAX_SEQUENCE_LENGTH integers
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # A 1D convolution with 128 output channels
    x = Conv1D(128, 5, activation='relu')(sequence_input)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    # A 1D convolution with 64 output channels
    x = Conv1D(64, 5, activation='relu')(x)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model

#=================================================================================================================#
# Image identifications Utils 
#=================================================================================================================#

def img_cnn(X, y):
    ''' Create a Convolutional layers based model for image classification
    X (ndarray): The features
    y (ndarray): The labels 
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    
    # input: a batch of images
    sequence_input = Input(shape=(90, 130), dtype='float32')
    
    # A 1D convolution with 128 output channels
    x = Conv1D(128, 5, activation='relu')(sequence_input)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    # A 1D convolution with 64 output channels
    x = Conv1D(64, 5, activation='relu')(x)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


#=================================================================================================================#
# Image and FCM values network utils
#=================================================================================================================#
 # Model not tested
def mixed_network(pulse_values, img, y, dp = 0.2):
    ''' Create a Mixed type Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    
    # Handling the values stemming from the Pulse files
    max_len = nb_curves = pulse_values.shape[1]
    nb_curves = pulse_values.shape[2]
    
    pulse_values_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    average = GlobalAveragePooling1D()(pulse_values_input)
    dense1 = Dense(64, activation='relu')(average)
    drop1 = Dropout(dp)(dense1)
    dense2 = Dense(32, activation='relu')(drop1)
    drop2 = Dropout(dp)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(dp)(dense3)
    dense4 = Dense(16, activation='relu')(drop3)
    drop4 = Dropout(dp)(dense4)
    
    # Handling the images (to check)
    img_height = img.shape[1]
    img_width = img.shape[2]
    img_input = Input(shape=(img_height, img_width), dtype='float32')
    
    # A level of 5 convolutional layers
    conv_1 = Conv1D(32, (5,), input_shape=(None, img_height, img_width), padding='same', activation = 'relu')(img_input)  
        
    # Concatenate: The parameters deriving from the 5 curves of concatenated
    concat = Concatenate()([conv_1, drop4])
    
    dense1 = Dense(64, activation='relu')(concat)
    avgpool1 = GlobalAveragePooling1D()(dense1)
    predictions = Dense(N_CLASSES, activation='softmax')(avgpool1)
    
    model = Model(inputs=[pulse_values_input, img_input], outputs=predictions)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model


#===============================================================================
# General Keras plotting utility 
#===============================================================================

def plot_losses(history):
    ''' Plot the train and valid losses coming from the training of the model 
    history (Keras history): The history of the model while training
    ----------------------------------------------------------------
    returns (plt plot): The train and valid losses of the model through the epochs
    '''
    # list all data in history
    #print(history.history.keys())
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