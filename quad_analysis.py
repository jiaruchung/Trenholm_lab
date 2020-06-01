# -*- coding: utf-8 -*-
"""
Created on Fri May 29 05:01:58 2020

@author: Kadjita Asumbisa and Jia-Ru Chang
"""
import glob
import numpy as np
import pandas as pd
import os


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


def quad_analysis(filespath,framerate=30):
    """
    Args: filepaths: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2) 
        files in filepaths must be name in the following format (animalId_sex_objectLocation_condition_.xlsx)
        i.e 0_m_ur_exp_.xlsx 
        framerate: an int or float of the capture frame rate of the camera in Hz. i.e 30 
    
    Output: 
        Data_quadrants: writes quad_coverage(aniaml id, condition, object location and coverage distance(mm), duration(s) and 
        velocity(mm/s) of all animals for each quadrant)to csv file in filepaths dir

    Returns:
        Saved Outputin a pandas dataframe  
    """    

    files,ids=get_files_info(filespath) #calls external fxn to get files info

    #creating pandas dataframe to hold data
    quad_coverage=pd.DataFrame(index=ids,columns=['cond','sex','object_loc','ul_dist', 'ul_dur',\
                       'ul_vel','ur_dist', 'ur_dur','ur_vel', \
                              'bl_dist', 'bl_dur','bl_vel','br_dist', 'br_dur','br_vel']) 
    
    for idx,file in enumerate(files): 
        mouse_id=int(file.split('\\')[-1].split('_')[0])
        sex=file.split('\\')[-1].split('_')[1]
        obj_loc=file.split('\\')[-1].split('_')[2]
        condition=file.split('\\')[-1].split('_')[3]
        
        
        #Assigning extracted data to be stored in their respective locs in the dataframe on line 44
        quad_coverage.loc[mouse_id,'cond']=condition
        quad_coverage.loc[mouse_id,'sex']=sex
        quad_coverage.loc[mouse_id,'object_loc']=obj_loc
        
        position_data=pd.read_excel(file) #reading psoition data into pd dataframe
        animal_pos=position_data.iloc[:,:2].dropna() #extracting first two cols and droping all nans
        
        #converting dataframes into np.arrays for pos x and y 
        pos_x=np.array(animal_pos.iloc[:,0])
        pos_y=np.array(animal_pos.iloc[:,1])
        
        #Defining center of the environment
        #Note: you may have to manually define the center of your environment for data with irregular path plots
        x_cen= (pos_x.max()+pos_x.min())/2 
        y_cen=(pos_y.max()+pos_y.min())/2
        
        #DISTANCE TRAVELLED
        #computes the change in pos_x and pos_y 
        dx = pos_x[1:]-pos_x[:-1] 
        dy = pos_y[1:]-pos_y[:-1]
        dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points and concats 0 to start.
        
        computed_vals=[] #empty list for holding computed total distance, total time and velocity in each quadrant
        for i in range(4):
            #Logical indexing for identifying differemt quadrants:
            if i==0:
                quad=(x_cen>pos_x) & (y_cen<pos_y)
            elif i==1:
                quad=(x_cen<pos_x) & (y_cen<pos_y)
            elif i==2:
                quad=(x_cen>pos_x) & (y_cen>pos_y)
            else:
                quad=(x_cen< pos_x) & (y_cen>pos_y)
                
            all_dist=dist[quad]   #using identified logical indexing to extract distances in specific quadrants 
           
            #Computing total distance, total time and velocity for current quadrant based on established formula  
            tot_dist=sum(all_dist)
            tot_time=len(all_dist)/framerate 
            vel=tot_dist/tot_time   
            
            computed_vals.extend([tot_dist,tot_time,vel]) #populating computed_vals for each iteration
            
        quad_coverage.iloc[mouse_id,3:]=computed_vals #assigning computed_vals to current mouse_id and respective cols in dataframe on line44
        
    quad_coverage.to_csv(filespath+'\Data_quadrants.csv', index_label='mouse_id') #write data to filespath as csv
    
    return quad_coverage

filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master'

quad_analysis(filespath)       


