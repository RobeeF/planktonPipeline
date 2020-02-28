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
import re

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
from pred_functions import predict
from keras.models import load_model
from time import time

start = time()

folder = 'C:/Users/rfuchs/Documents/cyto_classif'
file = 'FUMSECK_L2_fp/Labelled_Pulse6_2019-05-06 10h09.parq'


date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
pred_file = re.search(date_regex, file).group(1) + '.csv'
os.chdir(folder)

# Load pre-trained model
#LottyNet = load_model('LottyNet_FUMSECK') 
#model = LottyNet

# Load nomenclature
#tn = pd.read_csv('train_test_nomenclature.csv')
#tn.columns = ['Label', 'id']

# Making formated predictions 
source_path = folder + '/' + file
dest_folder = folder
predict(source_path, folder, LottyNet, tn, scale = False, pad = False)

# Getting those predictions
preds = pd.read_csv(folder + '/' + pred_file)

np.mean(preds['True FFT id'] == preds['Pred FFT id'])

colors = ['#96ceb4', '#ffeead', '#ffcc5c', '#ff6f69', '#588c7e', '#f2e394', '#f2ae72', '#d96459']

# True vs pred

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
for id_, label in enumerate(list(tn['Label'])):
    obs = preds[preds['True FFT Label'] == label]
    ax1.scatter(obs['Total FLO'], obs['Total FLR'], c = colors[id_], \
            label= label)
    ax1.legend(loc= 'lower right', shadow=True, fancybox=True, prop={'size':10})

ax1.set_title('True : Total FLO vs Total FLR')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1, 10**6)
ax1.set_ylim(1, 10**6)


for id_, label in enumerate(list(tn['Label'])):
    obs = preds[preds['Pred FFT Label'] == label]
    ax2.scatter(obs['Total FLO'], obs['Total FLR'], c = colors[id_], \
            label= label)
    ax2.legend(loc= 'lower right', shadow=True, fancybox=True, prop={'size':10})
ax2.set_title('Pred: Total FLO vs Total FLR')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1, 10**6)
ax2.set_ylim(1, 10**6)

end = time()
print(end - start)


####################### Three 2D projections in a row ################################
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.suptitle('2D cytograms')
ax1.scatter(preds['Total FLO'], preds['Total FLR'], c = preds['FFT id'])
ax1.set_title('Total FLO vs Total FLR')

ax2.scatter(preds['Total FWS'], preds['Total FLR'], c = preds['FFT id'])
ax2.set_title('Total FWS vs Total FLR')

ax3.scatter(preds['Total SWS'], preds['Total FLR'], c = preds['FFT id'])
ax2.set_title('Total SWS vs Total FLR')

fig.tight_layout()
fig.show()

#plt.scatter(preds['Total FLO'], preds['Total FLR'], c = preds['Particle_class_num'])
#plt.scatter(preds['Total FWS'], preds['Total FLR'], c = preds['Particle_class_num'])
#plt.scatter(preds['Total SWS'], preds['Total FLR'], c = preds['Particle_class_num'])





a = np.array(pd.DataFrame(true_labels, columns = ['Pred FFT Label']).merge(tn)['Pred FFT id'])
b = np.argmax(y_test, axis = 1)

print((a - b)[:1000])

true_labels[0]
np.argmax(y_test[0])