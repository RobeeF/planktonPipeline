# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""
import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from from_cytoclus_to_curves_values import extract_curves_values, format_for_pred
from time import time

##################################################################################################
# FUMSECK
##################################################################################################

# Extract the FLR 6
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 6
extract_curves_values(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


# Extract the FLR 25
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 25
extract_curves_values(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


##################################################################################################
# ENDOUME
##################################################################################################

data_source = 'C:/Users/rfuchs/Documents/SSLAMMP2-defaultpulseshapes'
data_destination = 'C:/Users/rfuchs/Documents/SSLAMMP2_L2'
flr_num = 25

start = time()
extract_curves_values(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)
end = time()
print('Time to extract', end - start)
# 6 s to extract 

format_for_pred(data_source, data_destination, flr_num = flr_num)
