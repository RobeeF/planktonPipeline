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

os.chdir('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values')

# Descriptive statistics
# Load training data
tn = pd.read_csv('train_nomenclature.csv')
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


# Visualize the predictions

preds = pd.read_csv('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values/FLR25 2019-12-06 06h09.csv')


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.suptitle('2D cytograms')
ax1.scatter(preds['Total FLO'], preds['Total FLR'], c = preds['Particle_class_num'])
ax1.set_title('Total FLO vs Total FLR')

ax2.scatter(preds['Total FWS'], preds['Total FLR'], c = preds['Particle_class_num'])
ax2.set_title('Total FWS vs Total FLR')

ax3.scatter(preds['Total SWS'], preds['Total FLR'], c = preds['Particle_class_num'])
ax2.set_title('Total SWS vs Total FLR')

fig.tight_layout()
fig.show()

#plt.scatter(preds['Total FLO'], preds['Total FLR'], c = preds['Particle_class_num'])
#plt.scatter(preds['Total FWS'], preds['Total FLR'], c = preds['Particle_class_num'])
#plt.scatter(preds['Total SWS'], preds['Total FLR'], c = preds['Particle_class_num'])
