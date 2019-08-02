# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:44:57 2019

@author: robin
"""

import os
from random import sample
from shutil import copy
import numpy as np


def partition_title_list(list_in, tronc = 0, is_test_set = False):
    ''' Partition the curves files names into train, test and vaidation sets. Be careful that it shuffles the original list
    list_in (list): The original list of files names to partition
    tronc (int): If positive keep only <tronc> elements of the list
    is_test_set (bool): whether or not to create a test set
    ------------------------------------------------------------------------
    returns (list of list): list_in partitioned in n random lists
    '''
    assert type(list_in) == list
    
    n = 4 # Split the list in four parts: the first two parts (half of the files names) constitute the training set. The other two parts represent the valid and train set
    
    tronc_list = sample(list_in, tronc) if tronc > 0 else list_in
    while len(tronc_list) % n != 0: # To avoid lists of different size pad the sequences with 'None' strings
        tronc_list.append('None')
    four_splitted_list = [tronc_list[i::n] for i in range(n)] 
    files_partition = {}
    if is_test_set:
        files_partition['train'] = four_splitted_list[0] + four_splitted_list[1]
        files_partition['test'] = four_splitted_list[2]
        files_partition['valid'] = four_splitted_list[3]
    else:
        files_partition['train'] = four_splitted_list[0] + four_splitted_list[1] + four_splitted_list[2]
        files_partition['test'] = []
        files_partition['valid'] = four_splitted_list[3]
    return files_partition

    
def imgs_train_test_valid_split(root_dir, balanced = False, is_test_set = False):
    ''' Split the curves contained in the <all> subdirectory of the <root_directory> into 3 folders: train, test and validation folders
    For the moment, I sample the same number of curves for each class (equal to the lowest umber of curves available)
    root_dir (str): The path to the folder containing the 4 folders: all, train, test and validation set (If everithing is ok root_directory is the 'curves' directory)
    ---------------------------------------------------------------------------------------------------------------
    returns (None): The data rightly split in three sub-datasets stored on hard disk
    '''
    curves_types = [f for f in os.listdir(root_dir + '/all')]

    if balanced:
        min_nb_curves = min([len(os.listdir(root_dir + '/all/' + ctype + '/' + cclass)) for ctype in curves_types for cclass in os.listdir(root_dir + '/all/' + ctype)])    
        min_nb_curves -=  min_nb_curves % 4  # Ensures that the all the plankton types have the same number of observations
        
    else: # Extract all available curves
        min_nb_curves = 0 #np.inf
        
    for ctype in curves_types:
        print(ctype)
        cluster_classes = [f for f in os.listdir(root_dir + '/all/' + ctype)]
        for cclass in cluster_classes:
            curves_files = [f for f in os.listdir(root_dir + '/all/' + ctype + '/' + cclass)]

            files_partition = partition_title_list(curves_files, min_nb_curves, is_test_set)
            for folder, files in files_partition.items():
                for file in files:
                    if file != 'None': 
                        copy(root_dir + '/all/' + ctype + '/' + cclass + '/' + file, root_dir + '/' + folder + '/' + ctype + '/' + cclass + '/' + file)
                        
def nb_available_imgs(root_dir):
    ''' Compute the number of available images in the folders '''
    # Assuming that all 5 ctypes repos have the same number of observations
    nb_train = np.sum([len(os.listdir(root_dir + '/train/Curvature/' + cclass)) for cclass in os.listdir(root_dir + '/train/Curvature')])    
    nb_test = np.sum([len(os.listdir(root_dir + '/test/Curvature/' + cclass)) for cclass in os.listdir(root_dir + '/test/Curvature')])    
    nb_valid = np.sum([len(os.listdir(root_dir + '/valid/Curvature/' + cclass)) for cclass in os.listdir(root_dir + '/valid/Curvature')])    

    return nb_train, nb_test, nb_valid

# Write a spec to ensure ok to fit NN on the curves: Same number of pic in each ctype subfolders 