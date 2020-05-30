# -*- coding: utf-8 -*-
"""
Created on Fri May 29 05:01:58 2020

@author: Kadjita Asumbisa and Jia-Ru Chang
"""
import glob
import numpy as np
import pandas as pd
import os

def quad_analysis(filespath,framerate=30):
    """
    Args: filepaths: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2) 
        files in filepaths must be name in the following format (animalId_objectLocation_condition_.xlsx)
        i.e 0_ur_exp_.xlsx 
        framerate: an int or float of the capture frame rate of the camera in Hz. i.e 30 
    
    Output: 
        writes quad_coverage(aniaml id, condition, object location and coverage distance(mm), duration(s) and 
        velocity(mm/s) of all animals for each quadrant)to csv file in filepaths dir        
    """    
    
    files=glob.glob(os.path.join(filespath,'*.xlsx'))
    ids=[int(file.split('\\')[-1].split('_')[0]) for idx,file in enumerate(files)]

    quad_coverage=pd.DataFrame(index=ids,columns=['cond','object_loc','object_id','ul_dist', 'ul_dur',\
                       'ul_vel','ur_dist', 'ur_dur','ur_vel', \
                              'bl_dist', 'bl_dur','bl_vel','br_dist', 'br_dur','br_vel']) 
    
    for idx,file in enumerate(files): 
        mouse_id=int(file.split('\\')[-1].split('_')[0])
        obj_loc=file.split('\\')[-1].split('_')[1]
        condition=file.split('\\')[-1].split('_')[2]
        '''define obj id'''
        
        
        quad_coverage.loc[mouse_id,'object_id']=#comp or mouse
        quad_coverage.loc[mouse_id,'object_loc']=obj_loc
        quad_coverage.loc[mouse_id,'cond']=condition
 
        
        position_data=pd.read_excel(file) 
        animal_pos=position_data.iloc[:,:2].dropna()
    
    
        pos_x=np.array(animal_pos.iloc[:,0])
        pos_y=np.array(animal_pos.iloc[:,1])
        
        #you may have to manually define the center of your environment for data with irregular path plots
        x_cen= (pos_x.max()+pos_x.min())/2 
        y_cen=(pos_y.max()+pos_y.min())/2
        
        
        #DISTANCE TRAVELLED
        dx = pos_x[1:]-pos_x[:-1]
        dy = pos_y[1:]-pos_y[:-1]
        dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points
        

        

        computed_vals=[]
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
            
            computed_vals.extend([tot_dist,tot_time,vel])
            
        quad_coverage.iloc[mouse_id,3:]=computed_vals
        
    """
    To DO
    Add sex to vars
    """
    
    quad_coverage.to_csv(filespath+'\quad_dat.csv')



filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master'

quad_analysis(filespath)       


