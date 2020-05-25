# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:43:23 2020

@author: kasum
"""

from scipy import interpolate
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt


filename=r'C:\Users\kasum\Downloads\Trenholm_lab-master\pos_object.xlsx'

position_data=pd.read_excel(filename) 

animal_pos=position_data.iloc[:,:2] #Extract animal XY cordinates for all frames
obj_pos=position_data.iloc[0,2:] #Extracts all the XY fixed object coordinates from frame 1

#Extracting object pos for X and Y. Object pos are arranges in a series of X and Y
obj_x=[obj_pos[i] for i in range(0,len(obj_pos),2)] #Even i corresponds to X
obj_y=[obj_pos[i+1] for i in range(0,len(obj_pos),2)] #Odd i corresponds to Y

#Initializing counts for bouts and corresponding animal XY pos for each bout
obj_bouts=0
bouts_xpos=[]
bouts_ypos=[]

#Computing object approach bouts using euclidean distance
for i in range(len(obj_x)): #iterate through number of object xy pair counts
    for j in range(len(animal_pos)): 
        dis=distance.euclidean([obj_x[i],obj_y[i]], [animal_pos.iloc[j,0],animal_pos.iloc[j,1]])
        if dis< 60: #Threshold for counting bouts
            obj_bouts+=1
            bouts_xpos.append(animal_pos.iloc[j,0])
            bouts_ypos.append(animal_pos.iloc[j,1])


######## PLOTS #################################
#Trajectory plot
plt.figure()
path, =plt.plot(animal_pos['X'],animal_pos['Y'],c='grey') # ',' in var defnition allows us to store only 1st item(1d).This makes the handle 1d for easy grp legend calls 

#Object points  plot
for i in range(0,len(obj_pos),2):
    obj=plt.scatter(obj_pos[i],obj_pos[i+1],c='black')

#Bouts plot                                       
bout=plt.scatter(bouts_xpos,bouts_ypos,c='r')
plt.legend([path,obj,bout],['path','object','bout'],loc='best')
    
'''To Do
Use interpolation to get more points
#f=interpolate.interp1d(x,y,kind='linear')

'''