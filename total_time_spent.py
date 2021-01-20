# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:32:55 2021

@author: labuser
"""

import numpy as np
import pandas as pd
from pylab import *
import glob, os    
###################
#import and process csv files
#####################

#Get all csv files in dir
#file_dir=r'E:\mouse_never_seen_others\day1'
file=r'F:\Nathan\Ready_for_Python\x_obj0day1\con1\cleanData\Data_quadrants.csv' #make list of paths
position=pd.read_csv(file)

con1_array=[]

for i in range(10):

    if position.object_loc[i]=='ur': 
        con1_array.append(position.ur_dur.values[i]) 
    elif position.object_loc[i]=='ul': 
        con1_array.append(position.ul_dur.values[i])      
    elif position.object_loc[i]=='br': 
        con1_array.append(position.br_dur.values[i])      
    elif position.object_loc[i]=='bl': 
        con1_array.append(position.bl_dur.values[i])  

#return con1_array            


#os.chdir(r'E:\mouse_never_seen_others\day1\con2')
file2=r'F:\Nathan\Ready_for_Python\x_obj0day1\con2\cleanData\Data_quadrants.csv' #make list of paths
position2=pd.read_csv(file2)

con2_array=[]

for i in range(10):

    if position2.object_loc[i]=='ur': 
        con2_array.append(position2.ur_dur.values[i]) 
    elif position2.object_loc[i]=='ul': 
        con2_array.append(position2.ul_dur.values[i])      
    elif position2.object_loc[i]=='br': 
        con2_array.append(position2.br_dur.values[i])      
    elif position2.object_loc[i]=='bl': 
        con2_array.append(position2.bl_dur.values[i])  

#return con2_array


first = pd.DataFrame(con1_array, columns =['con1'])
second = pd.DataFrame(con2_array, columns =['con2'])
result = pd.concat([first, second], axis=1, join="inner")

#Saving new csv to file_dir
new_dir=r'F:\Nathan\Results'
file_name = "total_time_spent_obj1"
result.to_excel(new_dir+'/'+file_name+'.xlsx', index=False)

