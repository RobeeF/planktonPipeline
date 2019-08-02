import pandas as pd
import os 
import re
import numpy as np


def extract_curves_values(data_source, data_destination, flr_num = 6):
    ''' Create 5 images per particle of the files in the data_source repository. The images are placed in folders mentioning the label of the particle
    data_source (str): The path to the original source of data
    data_features (str) : The path to write the data to
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    ---------------------------------------------------------------------------------------------
    returns (None): Write the images on hard disk
    '''
    assert (flr_num == 6) or (flr_num == 25)
    
    files_title = [f for f in os.listdir(data_source)]
    # Keep only the interesting csv files
    flr_title = [f for f in files_title if re.search("^FUMSECK-FLR" + str(flr_num),f) and re.search("csv",f) ]

    pulse_titles_clus = [f for f in flr_title if  re.search("Pulse",f) and not(re.search("Default",f))]
    pulse_titles_default = [f for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)]


    # Defining the regex
    date_regex = "FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"
    pulse_regex = "_([a-zA-Z0-9 ()]+)_Pulses.csv"  

    dates = set([re.search(date_regex, f).group(1) for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)])
    cluster_classes = list(set([re.search(pulse_regex, cc).group(1) for cc in pulse_titles_clus]))
    cluster_classes += ['noise']
  

    ### Create the directories that will host the curves
    if not os.path.exists(data_destination + '/curves_val'):
        os.makedirs(data_destination + '/curves_val') 
    
    for date in dates: # For each sampling phase
        #date = list(dates)[0]
        print("Processing:", date)
        pulse_data = pd.DataFrame()
        # Get the info about each particule and its cluster label
        date_datasets_titles = [t for t in pulse_titles_clus if re.search(date, t)]
        
        # For each file
        for title in date_datasets_titles:
            #title = date_datasets_titles[0]
            print(title)
            
            try:
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
            df = df[df["Particle ID"] != 0] # Delete formatting zeros
                        
            # Add the date of the extraction
            df["date"] = date
            
            # Get the name of the cluster from the file name
            clus_name = re.search(pulse_regex, title).group(1) 
            df["cluster"] = clus_name
            
            df.set_index("Particle ID", inplace = True)

            pulse_data = pulse_data.append(df)

        
        ## Extract info of noise particles 
        title = [t for t in pulse_titles_default if re.search(date, t)][0] # Dirty
        print(title)
        
        try:
            df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
        except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
            df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
        
        df["date"] = date
        df = df[df["Particle ID"] != 0] # Delete formatting zeros
        
        existing_indices = pulse_data[pulse_data['date']==date].index

        df.set_index("Particle ID", inplace = True)
        noise_indices = list(set(df.index) - set(existing_indices)) # Determining the noise particles indices
        df = df.loc[noise_indices] # Keep only the noise particles 
    
        clus_name = "noise"
        df["cluster"] = clus_name
    
        pulse_data = pulse_data.append(df)
        
        pulse_data.to_csv(data_destination + '/curves_val/Labelled_Pulse' + str(flr_num) + '_' + date + '.csv') # Store the data on hard disk