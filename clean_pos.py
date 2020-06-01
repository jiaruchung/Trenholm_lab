# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 02:43:20 2020

@author: Kadjita Asumbisa and Jia-Ru Chang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import glob

def get_files_info(filespath):
    """
    Args: filepaths: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2)
    
    Returns:
        files: paths to all .xlsx files in filespath dir
        ids: index of each animal_id stored as a list  
    """
    files=glob.glob(os.path.join(filespath,'*.xlsx')) #storing the dir of all .xlsx files input dir
    ids=[int(file.split('\\')[-1].split('_')[0]) for idx,file in enumerate(files)] #Extracting animal_ids to create index on line 37
    return files,ids



def clean_pos_data(raw_filespath, clean_filespath, thres=.95):
    """
    Args: raw_filespath: directory containing .xlsx raw data files
        clean_filespath: directory to write processed files to
        thres: threshold for dermining the distribution of data to keep during cleaning
        
    Outputs:
        figs: showing the raw and processed path plots for each animal 
        animal_pos: processed (clean) .xlsx version of each file saved to clean_filespath dir with original filename      
    """
    files,_=get_files_info(raw_filespath) #function call to return only first return variable
    
    for idx, file in enumerate(files):
        position_data=pd.read_excel(file) #read_excel file with position data
        animal_pos=position_data.iloc[:,:2]
    
        dx = np.array(animal_pos.iloc[:,0][1:])-np.array(animal_pos.iloc[:,0][:-1]); 
        dy = np.array(animal_pos.iloc[:,1][1:])-np.array(animal_pos.iloc[:,1][:-1]); 
    
        plt.figure()
        #### Raw Data Plot#################
        plt.suptitle('Mouse_'+file.split('\\')[-1].split('_')[0]+' Path Plots')
        ax=plt.subplot(211) 
        ax.set_title('Raw_data')      
        ax.plot(animal_pos.iloc[:,0],animal_pos.iloc[:,1])
        ax.set_xticks([])
        ax.set_yticks([])
           
        hist_counts,hist_bins=np.histogram(abs(dx))
        counts=0
        skip_x=True
        for i in range(len(hist_counts)):
            if counts <= len(dx)*thres:
                counts+=hist_counts[i]
                id_x=i
                threshold=hist_bins[id_x+1]
                skip_x=False
        
        #if no jumps in tracking, enter this loop        
        if not skip_x:  #if it is not true run the loop
            for i in range(len(dx)):
                if abs(dx[i])>threshold: #Tracking noise threshold
                    animal_pos.loc[i]= np.NaN
        
        #Since we are using distance to determine cut offs, we can use any axis and extend to the other
                if abs(dy[i])>threshold: #Here we are using threshold determined from x on line 62
                    animal_pos.loc[i]= np.NaN
     
        #### Processed Data plot##############        
        ax2=plt.subplot(212)       
        ax2.plot(animal_pos.iloc[:,0],animal_pos.iloc[:,1])
        ax2.set_title('Processed_data')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        animal_pos.dropna().reset_index(drop=True)# resetting may be redundant bcos of next line check on it 
        animal_pos.to_excel(clean_filespath+str('\\')+ file.split('\\')[-1], index=False)
         
raw_filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master\dirty' #filepath
clean_filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master\dirty\cl' #filepath
clean_pos_data(raw_filespath,clean_filespath)
