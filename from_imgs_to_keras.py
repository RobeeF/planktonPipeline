# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:44:57 2019

@author: robin
"""

import numpy as np 
import os
from random import sample
from shutil import copy


def partition_list(list_in, n, tronc = 0):
    ''' Partition a list into n random lists. Be careful that it shuffles the original list
    list_in (list): The original list to partition
    n (int): The number of parts to draw from list_in 
    tronc (int): If positive keep only <tronc> elements of the list
    ------------------------------------------------------------------------
    returns (list of list): list_in partitioned in n random lists
    '''
    assert type(list_in) == list
    
    tronc_list = sample(list_in, tronc) if tronc > 0 else list_in
    while len(tronc_list) % 3 != 0: # To avoid lists of different size pad the sequences with 'None' strings
        tronc_list.append('None')
    return [tronc_list[i::n] for i in range(n)]

    
def imgs_train_test_valid_split(root_dir):
    ''' Split the curves contained in the <all> subdirectory of the <root_directory> into 3 folders: train, test and validation folders
    For the moment, I sample the same number of curves for each class (equal to the lowest umber of curves available)
    root_dir (str): The path to the folder containing the 4 folders: all, train, test and validation set (If everithing is ok root_directory is the 'curves' directory)
    ---------------------------------------------------------------------------------------------------------------
    returns (None): The data rightly split in three sub-datasets stored on hard disk
    '''
    
    curves_types = [f for f in os.listdir(root_dir + '/all')]
    min_nb_curves = min([len(os.listdir(root_dir + '/all/' + ctype + '/' + cclass)) for ctype in curves_types for cclass in os.listdir(root_dir + '/all/' + ctype)])    
    min_nb_curves -=  min_nb_curves % 3  # Ensures that the all the plankton types have the same number of observations

    for ctype in curves_types:
        cluster_classes = [f for f in os.listdir(root_dir + '/all/' + ctype)]
        for cclass in cluster_classes:
            curves_files = [f for f in os.listdir(root_dir + '/all/' + ctype + '/' + cclass)]
            part_curves = np.stack(partition_list(curves_files, 3, min_nb_curves)).T
            folders = ['train', 'test', 'valid']
            for curve in part_curves:
                for idx, folder in enumerate(folders): 
                    if curve[idx] != 'None': 
                        copy(root_dir + '/all/' + ctype + '/' + cclass + '/' + curve[idx], root_dir + '/' + folder + '/' + ctype + '/' + cclass + '/' + curve[idx])