# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:36:29 2020

@author: kasum
"""

import numpy as np
import pandas as pd
from pylab import *
import glob, os    
###################
#import and process csv files
#####################

#Get all csv files in dir
file_dir=r'E:\Nathan\Raw_mouse_recordings\obj4-day1'
all_files = glob.glob(os.path.join(file_dir, "*.csv")) #make list of paths

#Create directory to store extracted position files
new_dir='/ready_for_python' # create this dir
try:
    os.mkdir(file_dir+new_dir)
except:
    pass
  
for file in all_files:
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0].split('DLC')[0]
    # Reading the file content to create a DataFrame
    pos = pd.read_csv(file)
    new_csv=pd.DataFrame(pos.iloc[2:,[1,2]].values)
    new_csv.columns=['X','Y']
    #Saving new csv to file_dir
    new_csv.to_excel(file_dir+new_dir+'/'+file_name+'.xlsx', index=False)
