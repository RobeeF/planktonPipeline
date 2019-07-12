# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:03:04 2019

@author: robin
"""

from keras.layers import Input, Dense, Conv1D, Concatenate, GlobalAveragePooling1D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


def toy_model(img_height = 200, img_width = 300):
    ''' First simple Keras model to test the approach
    img_height (int): Height of the 5 curves images given as inputs
    img_width (int): Width of the 5 curves images given as inputs    
    ------------------------------------------------------------------------
    returns (Keras model): The callabel Toy Model
    '''
    # Inputs: 5 curves
    img_1 = Input(shape=(img_height, img_width))
    img_2 = Input(shape=(img_height, img_width))
    img_3 = Input(shape=(img_height, img_width))
    img_4 = Input(shape=(img_height, img_width))
    img_5 = Input(shape=(img_height, img_width))
    
    # A level of 5 convolutional layers
    conv_1_1 = Conv1D(16, (5,), input_shape=(None, img_height, img_width), padding='same')(img_1)  
    conv_2_1 = Conv1D(16, (5,), input_shape=(None, img_height, img_width), padding='same')(img_2)  
    conv_3_1 = Conv1D(16, (5,), input_shape=(None, img_height, img_width), padding='same')(img_3)  
    conv_4_1 = Conv1D(16, (5,), input_shape=(None, img_height, img_width), padding='same')(img_4)  
    conv_5_1 = Conv1D(16, (5,), input_shape=(None, img_height, img_width), padding='same')(img_5)  
        
    # Concatenate: The parameters deriving from the 5 curves of concatenated
    concat = Concatenate()([conv_1_1, conv_2_1, conv_3_1, conv_4_1, conv_5_1])
    
    dense1 = Dense(16, activation='sigmoid')(concat)
    avgpool1 = GlobalAveragePooling1D()(dense1)
    predictions = Dense(7, activation='softmax')(avgpool1)
    
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
        
def fit_gen(gens_dict):
    # Batches of images are successively sent to the network
    img_batches = {}    
    while True:
        for curve_type in gens_dict.keys():         
            img_batches[curve_type] = gens_dict[curve_type].next()
        imgs_labels = list(img_batches.values()) # A bunch of batch_size * 5 curves and their label
        imgs = [imgs_labels[i][0][:,:,:,0] for i in range(len(imgs_labels))] # Extracting the curve images.

        labels = [imgs_labels[i][1] for i in range(len(imgs_labels))] # Extracting the label corresponding to the set of 5 curves
        assert all([np.mean(label == labels[0]) == 1.0 for label in labels]) # Checking that the 5 curves extracted belong to the same group for each observation
        yield imgs, labels[0] # The label is the same for the 5 curves, so return only the label of the first curve
        
def plot_losses(history):
    ''' Plot the train and valid losses coming from the training of the model '''
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