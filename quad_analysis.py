# -*- coding: utf-8 -*-
"""
Created on Fri May 29 05:01:58 2020

@author: Kadjita Asumbisa & Jia-Ru Chang
"""

import numpy as np
import pandas as pd

filename=r'C:\Users\kasum\Downloads\Trenholm_lab-master\data.xlsx' #filepath

def quad_analysis(filename,framerate=30):
    position_data=pd.read_excel(filename) #read_excel file with position data
    animal_pos=position_data.iloc[:,:2]
    
    
    pos_x=np.array(animal_pos.iloc[:,0])
    pos_y=np.array(animal_pos.iloc[:,1])
    
    #you may have to manually define the center of your environment for data with irregular path plots
    x_cen= (pos_x.max()+pos_x.min())/2 
    y_cen=(pos_y.max()+pos_y.min())/2
    
    
    #DISTANCE TRAVELLED
    dx = pos_x[1:]-pos_x[:-1]
    dy = pos_y[1:]-pos_y[:-1]
    dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points
    
    quad_coverage=pd.DataFrame(index=['up_left','up_right','buttom_left','buttom_right'], \
                 columns=['tot_dist','time_spent','velocity(unit)'])
    
    
    for i in range(4):
        if i==0:
            quad=(x_cen>pos_x) & (y_cen<pos_y)
        elif i==1:
            quad=(x_cen<pos_x) & (y_cen<pos_y)
        elif i==2:
            quad=(x_cen>pos_x) & (y_cen>pos_y)
        else:
            quad=(x_cen< pos_x) & (y_cen>pos_y)
            
        all_dist=dist[quad]        
        tot_dist=sum(all_dist)
        tot_time=len(all_dist)/framerate 
        vel=tot_dist/tot_time   
        
        quad_coverage.iloc[i,0]=tot_dist #Extracting distance covered in quadrant
        quad_coverage.iloc[i,1]=tot_time #Extracting distance covered in quadrant
        quad_coverage.iloc[i,2]=vel #Extracting distance covered in quadrant
    '''
    To Do: merge line 45-52
    '''
    return quad_coverage

quad_analysis(filename)

