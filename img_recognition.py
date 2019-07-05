# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:03:04 2019

@author: robin
"""

from keras.layers import Input, Dense, Conv1D, Concatenate, GlobalAveragePooling1D
from keras.models import Model

def toy_model(img_height = 200, img_width = 300):
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


##### Defining multi-input images generators 
# Generator for the 5 input
def multi_input_data_generator(source_generator, repo, batch_size, img_height = 200, img_width = 300):
    ''' Output the curves from the classes available in the <repo> directory 
    source_generator (ImageDataGenerator): The base generator defining the settings of the generation
    repo (str): Can be 'train', 'test' or 'valid'
    batch_size (int): The size of the batches to feed the neural network with
    ---------------------------------------------------------------------------
    returns (tuple): A batch of 5 curves samples and their label
    '''
    
    gens_dict = {} # The generator for each curve type will be holden in a dict
    curves_types = ['Curvature','FWS','SWS','FL Orange','FL Red']
    for curve_type in curves_types: 
        gen = source_generator.flow_from_directory(
            directory= './' + repo + '/' + curve_type + '/',
            target_size=(img_height, img_width),
            color_mode="grayscale",
            batch_size = batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        gens_dict[curve_type] = gen
    
    # Batches of images are successively sent to the network
    img_batches = {}    
    while True:
        for curve_type in curves_types:         
            img_batches[curve_type] = gens_dict[curve_type].next()
        imgs_labels = list(img_batches.values()) # A bunch of batch_size * 5 curves and their label
        imgs = [imgs_labels[i][0][:,:,:,0] for i in range(len(imgs_labels))] # Extracting the curve images.
        labels = [imgs_labels[i][1] for i in range(len(imgs_labels))] # Extracting the label corresponding to the set of 5 curves
        yield imgs, labels[0] # The label is the same for the 5 curves, so return only the label of the first curve