import os
import numpy as np
import re
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from from_cytoclus_to_curves_values import extract_curves_values
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

    
def get_curves_values_data(raw_source, clean_source, cluster_classes, extract_curves = False, pad = True, seed = None):
    ''' Generate a balanced dataset from the Pulse files 
    raw_source (str): The location of raw Pulse files (if extraction is needed)
    clean_source (str): The location of extracted (and formatted) Pulse files on disk
    cluster_classes (list of str): The classes used in the prediction task
    extract_curves (Bool): Whether to proceed to the extraction of the data (set it to True the first time you run the script)
    pad (Bool): Whether to pad (resp. truncate) the short sequence (resp.the long sequence) or to interpolate them. Default is to pad
    
    ------------------------------------------------------------------------------
    return (4 arrays): The dataset (X, y) the particle ids (pid) and the encoder of the groups names    
    '''
    if extract_curves:
        print('Extraction from raw data in progress')
        extract_curves_values(raw_source, clean_source, flr_num = 25) 
    
    le = LabelEncoder()
    le.fit(cluster_classes) 
    
    # Extracting the values
    path = clean_source 
    files = os.listdir(path)
    X = []
    y = []
    pid_list = []
    seq_len_list = []
    
    # Get the records of how many observations per class have already been included in the dataset
    nb_obs_to_extract_per_group = 100
    balancing_dict = dict(zip(cluster_classes, np.full(len(cluster_classes), nb_obs_to_extract_per_group)))
    
    for idx, file in enumerate(files):
        print('File:', idx + 1, '/', len(files), '(', file, ')')
        df = pd.read_csv(path + '/' + file)

	# Add arg in prototype
	#if spe_flr_extract_fft:
		#flr_num = re.search('(?:flr|FLR)([0-9]){2}').group(1) 
		#if flr_num <= 7:   	
			#df = df[(df['cluster'] != 'cryptophyte') & (df['cluster'] != 'nanoeucaryote') & (df['cluster'] != 'microphytoplancton')]
		#if flr_num >= 8:
			#df = df[(df['cluster'] != 'picoeucaryote') & (df['cluster'] != 'prochlorococcus') & (df['cluster'] != 'synechococcus')]
	
        X_file, seq_len_file, y_file, pid_list_file = data_preprocessing(df, balancing_dict, 120, pad = pad, seed = seed)

        X.append(X_file)
        y.append(y_file)
        pid_list.append(pid_list_file)
        seq_len_list.append(seq_len_file)
        
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
    seq_len_list = np.concatenate(seq_len_list)
        
    # Sanity check:
    assert len(X) == len(y) == len(pid_list) == len(seq_len_list)
        
    return X, seq_len_list, y, pid_list, le

    
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
            
    return np.stack(x)


def interp_sequences(sequences, max_len):
    ''' Interpolate sequences in order to reduce their length to max_len
        sequences (ndarray): The sequences to interpolate
        maxlen (int): The maximum length of the sequence: shorter sequences will be padded with <value>, longer sequences will be truncated 
        -------------------------------------------------------------------
        returns (ndarray): The interpolated sequences
    '''

    interp_obs = []
    # Looping is dirty... Padding then interpolating and de-padding might be better
    for idx, s in enumerate(sequences): 
        original_len = s.shape[1]
        f = interp1d(np.arange(original_len), s, 'quadratic', axis = 1)
        interp_seq = np.apply_along_axis(f, 0, np.linspace(0, original_len -1, num = max_len))
        interp_obs.append(interp_seq)
    
    return np.stack(interp_obs) 


def data_preprocessing(df, balancing_dict, max_len = None, pad = False, seed = None):
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
    
    # If the group that have less observations is not represented in this dataset: sample 40 observations of the other groups
    if np.sum(list(balancing_dict.values())) == 0:
        balancing_dict = {k: min(40, clus_value_count[k]) for k in clus_value_count.keys()}
            
    # Undersampling to get a balanced dataset
    rus = RandomUnderSampler(random_state = seed, ratio = balancing_dict)
        
    pid_resampled, y_resampled = rus.fit_sample(pid_cluster['Particle ID'].values.reshape(-1,1), pid_cluster['cluster'])
    df_resampled = df.set_index('Particle ID').loc[pid_resampled.flatten()]
    
    # Reformatting the values
    obs_list = [] # The 5 curves
    pid_list = [] # The Particle ids
    y_list = [] # The class (e.g. 0 = prochlorocchoccus, 1= ...)
    seq_len_list = [] # The original length of the sequence
    
    for pid, obs in df_resampled.groupby('Particle ID'):
        # Sanity check: only one group for each particle
        assert(len(set(obs['cluster'])) == 1) 

        obs_list.append(obs.iloc[:,:5].values.T)
        seq_len_list.append(len(obs))
        pid_list.append(pid)
        y_list.append(list(set(obs['cluster']))[0])
    
    # Defining a fixed length for all the sequence: 0s are added for shorter sequences and longer sequences are truncated
    if max_len == None:
        max_len = int(np.percentile(seq_len_list, 75))
    
    if pad:
        obs_list = custom_pad_sequences(obs_list, max_len)
    else:
        obs_list = interp_sequences(obs_list, max_len)
    
    X = np.transpose(obs_list, (0, 2, 1))
    
    return X, seq_len_list, y_list, pid_list 

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

def scaler(X):
    ''' Scale the data. For the moment only minmax scaling is implemented'''
    X_mms = []

    # load data
    # create scaler
    scaler = MinMaxScaler()
    # fit and transform in one step
    for obs in X:
        normalized = scaler.fit_transform(obs)
        X_mms.append(normalized)
    
    X_mms = np.stack(X_mms)
    return X_mms


def extract_features_from_nn(dataset, pre_model):
    ''' Extract and flatten the output of a Neural Network
    dataset ((nb_obs,curve_length, nb_curves) array): The interpolated and scaled data
    pre_model (Keras model): The model without his head
    ---------------------------------------------------------------------
    returns ((nb_obs,nb_features) array): The features extracted from the NN
    '''
     
    features = pre_model.predict(dataset, batch_size=32)
    features_flatten = features.reshape((features.shape[0], -1))
    return features_flatten