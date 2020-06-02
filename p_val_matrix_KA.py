# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:29:32 2020

@author: Kadjita Asumbisa and Jia-Ru Chung 
"""
import numpy as np
import pandas as pd
import glob
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


def occu_matrix(animal_pos,bins=10):
    """
    Args: 
        animal_pos: pandas dataframe animal pos X and Y in cols 1 and 2 in the folder of two conditions 
        bins: bin size for determining size of matrix. Default set to 10
    Returns: 
        occu_mat: occupancy matrix of a 
    """
    #Extracting x and y pos and bin lims
    xpos=animal_pos.iloc[:,0]
    ypos=animal_pos.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
    occu_mat, _, _ = np.histogram2d(xpos, ypos, [xbins,ybins]) #creating a 2d matrix of xpos and ypos defined by bins sizes on line36&37
    occu_mat= occu_mat[::-1] #flips the matrix to align with original path plot
    return occu_mat


def align_allMats(filespath):
    """
    Args: 
        filespath: Path to all the .xlsx files with animal pos X and Y in cols 1 and 2 in the folder of two conditions 
    Returns: 
        alingned_to_ul: occupancy matrix of all animals in dir where object loc aligned to the upper-left quadrant 
    """ 
    
    #Ask user to confirm if filespath is set to folder with processed (clean) files
    clean_files=input('Is filespath set to the clean files directory?: (y/n) ').lower()
    if clean_files=='y':
        print('realigning all object positions to upper left quadrant...')
         
        files,_=get_files_info(filespath) #function call to return all file directories
        
        alingned_to_ul=[] #Initializing an empty list to hold realigned position matrix 
        for file in files:
            position=pd.read_excel(file) #reading position file into a pandas dataframe
            animal_pos=position.iloc[:,:2] #extracting posx and y which corresponds to cols 1&2
    
            object_loc=file.split('\\')[-1].split('_')[2] #Extracting object location
            
            #flip position matrix depending on object loc to ensure that all objects locs across animals are aligned to upper left quadrant 
            if object_loc=='ur': 
                ur_matrix=occu_matrix(animal_pos) #fxn call to compute occupancy matrix 
                flipped_from_ur=np.flip(ur_matrix,axis=1)
                alingned_to_ul.append(flipped_from_ur) 
            elif object_loc=='bl':
                bl_matrix=occu_matrix(animal_pos)
                flipped_from_bl=np.flip(bl_matrix,axis=0)
                alingned_to_ul.append(flipped_from_bl)
            elif object_loc=='br':
                br_matrix=occu_matrix(animal_pos)
                flipped_from_br=np.flip(br_matrix,axis=0) #flip on first axis
                flipped_from_br=np.flip(flipped_from_br,axis=1) #flip on second axis
                alingned_to_ul.append(flipped_from_br) #append the final flipped matrix
            else:
                ref_matrix=occu_matrix(animal_pos) #ul: arbitrarily pre-determined as reference object location for realignment 
                alingned_to_ul.append(ref_matrix)
        return alingned_to_ul
    else:
        print('set filespath to folder with clean excel files before running this function')


filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master'
data=align_allMats(filespath)

#Notes
#put experimental and control data in seperate directories
#if we want all to be in the same fucntion, then we need to set another loop

###########PLOTTING######################
def occu_heatmap(pos):  
    xpos=pos['X']
    ypos=pos['Y']
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, BINS+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, BINS+1)
    occu, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    occu=gaussian_filter(occu,sigma=0.7)
    fig,ax=plt.subplots()
    q = imshow(occu, cmap='jet', interpolation='bilinear')
    gca().set_yticks([])
    gca().set_xticks([])
    #ax.axis('off')
 #   cticks=cbar.ax.get_xticks()
 #   cbar.set_ticks([])
    min_=plt.text(11,9.5,'min')
    max_=plt.text(11,-0.3,'max')
    #ax.invert_yaxis()
    return q,occu
