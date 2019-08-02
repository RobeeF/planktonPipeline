from keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, MaxPooling1D, Conv1D, LSTM, Flatten
from keras.models import Model
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import os


from imblearn.under_sampling import RandomUnderSampler
import re
import pandas as pd

from extract_Pulse_values.from_cytoclus_to_curves_values import extract_curves_values
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


def get_encoder(path):
    ''' Create the encoder object to encode and decode the names of the groups into integers
    path (str): The path to where data are stored 
    -------------------------------------------------------------------------
    returns (label encoder): The label encoder object'''
    
    # Encode the names of the clusters into integers
    flr_titles = os.listdir('W:/Bureau/these/donnees_oceano/Newprocess_20190729_FLR25')
    
    if len(flr_titles) == 0:
        raise RuntimeError('The data_destination folder that you specified is empty. You might consider reextracting the curves')
        
    pulse_titles_clus = [f for f in flr_titles if  re.search("Pulse",f) and not(re.search("Default",f))]
    
    pulse_regex = "_([a-zA-Z0-9 ]+)_Pulses.csv"  
    cluster_classes = list(set([re.search(pulse_regex, cc).group(1) for cc in pulse_titles_clus \
                                if not(re.search('Listmode', cc))]))
    
    cluster_classes.append('noise')
    
    cluster_classes = homogeneous_cluster_names(cluster_classes) # Get homogeneous groups names between samples
    le = LabelEncoder()
    le.fit(cluster_classes) 
    
    return le
    
def get_curves_values_data(data_source, data_destination, extract_curves = False):
    ''' Generate a balanced dataset from the Pulse files 
    data_source (str): The location of the Pulse files
    data_destination (str): Where to write the formated Pulse files on disk
    extract_curves (Bool): Whether to proceed to the extraction of the data (set it to True the first time you run the script)
    ------------------------------------------------------------------------------
    return (4 arrays): The dataset (X, y) the particle ids (pid) and the encoder of the groups names    
    '''
    if extract_curves:
        extract_curves_values(data_source, data_destination, flr_num = 25) 

    # Encode the names of the clusters into integers
    le = get_encoder(data_source)  
    cluster_classes = le.classes_
    
    # Extracting the values
    path = data_destination + '/curves_val'
    files = os.listdir(path)
    X = []
    y = []
    pid_list = []
    
    nb_obs_to_extract_per_group = 100
    balancing_dict = dict(zip(cluster_classes, np.full(len(cluster_classes), nb_obs_to_extract_per_group)))
    
    for idx, file in enumerate(files):
        print('File:', idx + 1, '/', len(files), '(', file, ')')
        df = pd.read_csv(path + '/' + file)
    
        X_file, y_file, pid_list_file = data_preprocessing(df, balancing_dict, 120)
        X.append(X_file)
        y.append(y_file)
        pid_list.append(pid_list_file)
        
        # Defining the groups to sample in priority in the next sample: Those which have less observations
        balancing_dict = pd.Series(np.concatenate(y)).value_counts().to_dict()
        balancing_dict = {k: balancing_dict[k] if k in balancing_dict else 0 for k in cluster_classes} 
        balancing_dict = {k: max(balancing_dict.values()) - balancing_dict[k] for k in cluster_classes}
            
    # Give the final form to the dataset    
    X = np.vstack(X)
    y = np.concatenate(y)
    y = le.transform(y)
    y =  to_categorical(y)
    pid_list = np.concatenate(pid_list)
    
    assert len(X) == len(y) == len(pid_list)
        
    return X, y, pid_list, le    
    

def custom_pad_sequences(sequences, maxlen=None, dim=5, dtype='float32',
    padding='pre', truncating='pre', value=0.):
    ''' Override keras method to allow multiple feature dimensions (adapted from Stack Overflow discussions).
        sequences (ndarray): The sequences to pad/truncate
        maxlen (int): The maximum length of the sequence: shorter sequences will be padded with <value>, longer sequences will be truncated 
        dim (int): input feature dimension (number of features per timestep)
        dtype (str): The type of the data
        padding (either 'pre' or 'post'): Whether to pad at the beginning (pre) or at the end of the sequence (post)
        padding (either 'pre' or 'post'): Whether to truncate at the beginning (pre) or at the end of the sequence (post)
        value (float): The value used to pad the sequences
        ------------------------------------------------------------------------------------------------------------------
        returns ((nb of observations, dim, maxlen) array): The padded sequences

    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    
    x = (np.ones((nb_samples, dim, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[:, -maxlen:]
        elif truncating == 'post':
            trunc = s[:, :maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, : , :trunc.shape[1]] = trunc
        elif padding == 'pre':
            x[idx, :, -trunc.shape[1]:] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x



def data_preprocessing(df, balancing_dict, max_len = None, seed = None):
    ''' Add paddings to Pulse sequences and rebalance the dataset 
    df (pandas DataFrame): The data container
    balancing_dict (dict): A dict that contains the desired quantities to extract for each group in order to obtain a balanced dataset
    maxlen (int): The maximum length of the sequence: shorter sequences will be padded with zeros, longer sequences will be truncated 
    seed (int): The seed to use to fix random results if wanted
    -----------------------------------------------------------------------------------------------------------------------------------
    returns (3 arrays): The dataset (X, y_list) y_list being unencoded labels and pid_list the list of corresponding particle ids
    '''
    # Make the cluster names homogeneous and get the group of each particule
    df = homogeneous_cluster_names(df)
    pid_cluster = df[['Particle ID', 'cluster']].drop_duplicates()
    clus_value_count = pid_cluster.cluster.value_counts().to_dict()
        
    # Deleting non existing keys and adapting to the data in place
    balancing_dict  = {k: min(balancing_dict[k], clus_value_count[k]) for k in clus_value_count.keys()}
    
    # If the group that have less observations is not represented in this dataset: sample 10 observations of the other groups
    if np.sum(list(balancing_dict.values())) == 0:
        balancing_dict = {k: min(40, clus_value_count[k]) for k in clus_value_count.keys()}
            
    # Undersampling to get a balanced dataset
    rus = RandomUnderSampler(random_state = seed, ratio = balancing_dict)
        
    pid_resampled, y_resampled = rus.fit_sample(pid_cluster['Particle ID'].values.reshape(-1,1), pid_cluster['cluster'])
    df_resampled = df.set_index('Particle ID').loc[pid_resampled.flatten()]
    
    # Reformatting the values
    obs_list = []
    pid_list = []
    y_list = []
    for pid, obs in df_resampled.groupby('Particle ID'):
        obs_list.append(obs.iloc[:,:5].values.T)
        pid_list.append(pid)
        assert(len(set(obs['cluster'])) == 1) # Sanity check: only one group for each particle
        y_list.append(list(set(obs['cluster']))[0])
    
    # Defining a fixed length for all the sequence: 0s are added for shorter sequences and longer sequences are truncated
    sequence_len = [obs.shape[1] for obs in obs_list]
    if max_len == None:
        max_len = int(np.percentile(sequence_len, 75))
    
    obs_list = np.stack(custom_pad_sequences(obs_list, max_len))
    
    X = np.transpose(obs_list, (0, 2, 1))
    
    return X, y_list, pid_list

def homogeneous_cluster_names(array):
    ''' Make homogeneous the names of the groups coming from the different Pulse files
    array (list, numpy 1D array or dataframe): The container in which the names have to be changed
    -----------------------------------------------------------------------------------------------
    returns (array): The array with the name changed and the original shape  
        '''
    if type(array) == pd.core.frame.DataFrame:
        array['cluster'] = array.cluster.str.replace('coccolithophorideae like','nanoeucaryote')
        array['cluster'] = array.cluster.str.replace('PicoHIGHFLR','picoeucaryotes')
        array['cluster'] = array.cluster.str.replace('es$','e') # Put in the names in singular form
        array['cluster'] = array.cluster.str.lower()
        
    else:
        array = [re.sub('coccolithophorideae like','nanoeucaryote', string) for string in array]
        array = [re.sub('PicoHIGHFLR','picoeucaryotes', string) for string in array]
        array = [re.sub('es$','e', string) for string in array]
        array = [string.lower() for string in array]

        array = list(set(array))
                
    return array
    

#=====================================================================================================================
# Keras models for Pulse files values 
#=====================================================================================================================

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

def conv_model(X, y): # Overfit en 1 epoch
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