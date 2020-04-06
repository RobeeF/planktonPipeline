# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:06:49 2019

@author: Utilisateur
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
from copy import deepcopy
from sklearn.metrics import confusion_matrix

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

#################################################################################################################
# Descriptive statistics
#################################################################################################################

# Load training data
tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

X = np.load('X.npy')
y = np.load('y.npy')
pid_list = np.load('pid_list.npy')
seq_len_list = np.load('seq_len_list.npy')

labels = np.argmax(y, axis = 1)

# Compute the quantiles
qs = np.quantile(seq_len_list, q = [0, 0.25, 0.5, 0.75, 1])

q1_mask = (seq_len_list >= qs[0]) & (seq_len_list< qs[1])
q2_mask = (seq_len_list >= qs[1]) & (seq_len_list<= qs[2])
q3_mask = (seq_len_list > qs[2]) & (seq_len_list <= qs[3])
q4_mask = (seq_len_list > qs[3]) & (seq_len_list<= qs[4])

assert((q1_mask.sum() + q2_mask.sum() + q3_mask.sum() + q4_mask.sum()) == len(seq_len_list))

seq_quant = deepcopy(seq_len_list)
seq_quant[q1_mask] = 1
seq_quant[q2_mask] = 2
seq_quant[q3_mask] = 3
seq_quant[q4_mask] = 4

pd.crosstab(seq_quant, labels)

len_distrib = pd.DataFrame({'Particle ID': pid_list, 'length': seq_len_list, 'q': seq_quant, 'label': labels})
len_distrib = len_distrib.merge(tn)

for label, group in len_distrib.groupby('Particle_class'):
    print('Group number', label)
    plt.hist(group['length'])
    plt.show()

len_distrib[len_distrib['Particle_class'] == 'cryptophyte']

len_distrib[len_distrib['Particle ID'] == 592.0]

###################################################################################################################
# Visualize the predictions
###################################################################################################################
from pred_functions import predict, plot_2D
from keras.models import load_model
from time import time

start = time()

folder = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM/L3'
file = 'SSLAMM/L3/Labelled_Pulse6_2019-05-06 10h09.parq'


date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
pred_file = 'Pulse6_2019-09-18 14h35.csv'
os.chdir(folder)

# Load pre-trained model
LottyNet = load_model('LottyNet_FUMSECK') 
model = LottyNet

# Load nomenclature
tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Label', 'id']

# Making formated predictions 
source_path = folder + '/' + file
dest_folder = folder
predict(source_path, folder, LottyNet, tn, scale = False, pad = False)

# Getting those predictions
preds = pd.read_csv(folder + '/' + pred_file)

np.mean(preds['True FFT id'] == preds['Pred FFT id'])

colors = ['#96ceb4', '#ffeead', '#ffcc5c', '#ff6f69', '#588c7e', '#f2e394', '#f2ae72', '#d96459']

#####################
# 2D plots
#####################

plot_2D(preds, tn, 'Total FLO', 'Total FLR', loc = 'lower right') # FLO vs FLR
plot_2D(preds, tn, 'Total FWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FWS', loc = 'upper left')


####################
# Confusion matrix
####################

lab_tab = tn.set_index('id')['Label'].to_dict()
cluster_classes = list(lab_tab.values())
true = np.array(preds['True FFT id'])
pred_values = np.array(preds['Pred FFT id'])

preds['Pred FFT id'].value_counts()

pred_values[np.isnan(true)]
    
    

cm = confusion_matrix(preds['True FFT Label'], preds['Pred FFT Label'], cluster_classes)
cm = cm/cm.sum(axis = 1, keepdims = True)
cm = np.where(np.isnan(cm), 0, cm)
print(cm) 

fig = plt.figure(figsize = (16,16)) 
ax = fig.add_subplot(111) 
cax = ax.matshow(cm) 
plt.title('Confusion matrix of LottyNet_Full on a FLR6 file') 
fig.colorbar(cax) 
ax.set_xticklabels([''] + labels) 
ax.set_yticklabels([''] + labels) 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.show()


###################################################################################################################
# Randomly pick some Endoume predictions and plot true vs pred 
###################################################################################################################
import re
import fastparquet as fp
from ffnn_functions import homogeneous_cluster_names


true_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_True_L1'
pred_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_L2'
graph_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/graphs_true_pred/1_23_03_20'

nb_plots = 10
true_files = [f for f in os.listdir(true_folder) if re.search('parq', f)]
picked_true_files = np.random.choice(true_files, nb_plots, replace = False)

date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
titles = [re.sub('Pulse', 'FLR', re.search(date_regex, f).group(0)) for f in picked_true_files]
picked_pred_files = [re.search(date_regex, f).group(0) + '.csv' for f in picked_true_files]

acc = np.zeros(nb_plots)

for i, file in enumerate(picked_true_files):
    pfile = fp.ParquetFile(true_folder + '/' + file)
    true = pfile.to_pandas(columns=['Particle ID','cluster'])
    true = true.reset_index().drop_duplicates()
    true = homogeneous_cluster_names(true)
    true.columns = ['Particle ID', 'True FFT Label']
    
    pred = pd.read_csv(pred_folder + '/' + picked_pred_files[i])
    
    if len(pred) != len(true):
        raise RuntimeError('Problem on', file)
    
    true_pred = pd.merge(true, pred, on = 'Particle ID')
    acc[i] = np.mean(true_pred['True FFT Label'] == true_pred['Pred FFT Label'])
    print(acc[i])
    plot_2D(true_pred, tn, 'Total FWS', 'Total FLR', loc = 'upper left', title = graph_folder + '/' + titles[i])


np.mean(acc)
q1 = 'Total FWS'
q2 = 'Total FLR'
#fp.to_pandas
#pd.read()
#true_pred = pd.merge([true, preds, on = 'Particle id')
#np.mean(true_pred['True FFT id'] == true_pred['Pred FFT id'])
#plot_2D(preds, tn, 'Total FWS', 'Total FLR', loc = 'upper left') # Add savefig ? 
plt.savefig('cocuou')


L1_pred_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_L1'
pd.read_csv(L1_pred_folder + "/" + file)

pfile2 = fp.ParquetFile(true_folder + '/' + file)
a = pfile2.to_pandas()
    
a.loc[88.0]
true.loc[88.0]
set(a.index) - set(true.index)
set(true.index) - set(a.index)

#############################################################################################
# Plot predicted time series 
#############################################################################################
from ffnn_functions import homogeneous_cluster_names

ts = pd.read_csv('C:/Users/rfuchs/Documents/09_to_12_2019.csv')
ts['date'] =  pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
ts = ts.set_index('date')

cols_plot = ts.columns
axes = ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')
    

# Formatting True time series for P1
true_ts = pd.read_csv('C:/Users/rfuchs/Documents/09_to_12_2019_true.csv', sep = ';', engine = 'python')
true_ts = true_ts[['Date','count', 'set']]
true_ts['Date'] =  pd.to_datetime(true_ts['Date'], format='%d/%m/%Y %H:%M:%S')
true_ts.columns = ['Date','count', 'cluster']
true_ts = homogeneous_cluster_names(true_ts)
true_ts['cluster'] = true_ts['cluster'].replace('default (all)', 'noise')

true_ts = true_ts.set_index('Date')


for cluster_name in ts.columns:
    if cluster_name in set(true_ts['cluster']):
        # Picoeuk comparison: (HighFLR are neglected)
        true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == cluster_name]['count'])
        true_ts_clus.columns = ['true_count']
        pred_ts_clus = pd.DataFrame(ts[cluster_name])
        pred_ts_clus.columns = ['pred_count']
        
        true_ts_clus.index = true_ts_clus.index.floor('H')
        pred_ts_clus.index = pred_ts_clus.index.floor('H')
        
        all_clus = true_ts_clus.join(pred_ts_clus)
        
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')
    else:
        print(cluster_name, 'is not in true_ts pred')

ts['microphytoplancton'].plot()
