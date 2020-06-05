# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:43:23 2020

@author: Kadjita Asumbisa and Jia-Ru Chung
"""

from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import glob,os

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


def object_bouts(filespath, bout_distance=50, sample_fig=True, framerate=30):
    """
    Args: filepaths: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2)
        and fixed object XY positions in remaining columns i.e X1,Y1,X2,Y2,X3,Y3 
        files in filepaths must be named with the following format (animalId_sex_objectLocation_condition_.xlsx)
        i.e 0_m_ur_exp_.xlsx 
        bout_distance: an int or float corresponding to the threshold(distance). Default set to '50' mm
        sample_fig: boolean which shows an example fig using data from mouse0 if True. Default set to 'True'
        framerate: an int or float of the capture frame rate of the camera in Hz. Default set to '30' Hz
    
    Output: 
        fig: displays exapmple fig if sampe_fig is set to True
        Data_bouts.csv: writes bouts_data (aniaml_id, condition, sex, object_location, total_bouts and bouts_duration (s) 
        for all animals to csv file in filepaths dir

    Returns:
        Saved Data_bouts in a pandas dataframe
    """  

    files,ids=get_files_info(filespath) #calls external fxn to get files info
    
    #Creating an empty dataframe to store processed data
    bouts_data=pd.DataFrame(index=ids,columns=['cond','sex','object_loc','tot_bouts','bouts_dur']) 
    
    #Extracting xlsx filenames and index from filepaths folder for further processing
    for idx,file in enumerate(files): 
        mouse_id=int(file.split('\\')[-1].split('_')[0])
        sex=file.split('\\')[-1].split('_')[1]
        obj_loc=file.split('\\')[-1].split('_')[2]
        condition=file.split('\\')[-1].split('_')[3]

        #Assigning extracted data to be stored in their respective locs in the dataframe on line 36    
        bouts_data.loc[mouse_id,'cond']=condition
        bouts_data.loc[mouse_id,'sex']=sex
        bouts_data.loc[mouse_id,'object_loc']=obj_loc
        
        position_data=pd.read_excel(file)     #read each file into a pd dataframe
        animal_pos=position_data.iloc[:,:2].dropna() #Extract animal XY cordinates for all frames and removing rows with nans
        obj_pos=position_data.iloc[0,2:] #Extracts all the XY for fixed object coordinates from frame 1(row0)
        
        #Extracting object pos for X and Y. Object pos are arranges in a series of X and Y
        obj_x=[obj_pos[i] for i in range(0,len(obj_pos),2)] #Even i corresponds to X
        obj_y=[obj_pos[i+1] for i in range(0,len(obj_pos),2)] #Odd i corresponds to Y
        
        #Initializing counts for bouts and corresponding animal XY pos for each bout and storing corresponding pos
        obj_bouts=0
        bouts_xpos=[]
        bouts_ypos=[]
        
        #Computing object approach bouts using euclidean distance
        for i in range(len(obj_x)): #iterate through number of object xy pair counts
            for j in range(len(animal_pos)): 
                dist=distance.euclidean([obj_x[i],obj_y[i]], [animal_pos.iloc[j,0],animal_pos.iloc[j,1]]) #euclidean distance between frames for fixed object points and animal pos
                if dist <= bout_distance: #Threshold for counting bouts
                    obj_bouts+=1
                    bouts_xpos.append(animal_pos.iloc[j,0])
                    bouts_ypos.append(animal_pos.iloc[j,1])
        
        #Dividing the bout count by frame rate gives the totol duration of bouts in secs        
        bout_duration=obj_bouts/framerate
        
        #Storing computed values of total bouts and bouts duration for each mouse into dataframe         
        bouts_data.loc[mouse_id, 'tot_bouts']=obj_bouts
        bouts_data.loc[mouse_id, 'bouts_dur']=bout_duration
    
        ######## EXAMPLE FIG #################################
        if sample_fig and mouse_id==0: 
            #Trajectory plot
            fig,ax=plt.subplots()
            path, =plt.plot(animal_pos.iloc[:,0].values,animal_pos.iloc[:,1].values,c='grey') # ',' in var defnition allows us to store only 1st item(1d).This makes the handle 1d for easy grp legend calls 
            
            #Object points plot
            for i in range(0,len(obj_pos),2):
                obj=ax.scatter(obj_pos[i],obj_pos[i+1],c='black')
            
            #Bouts plot                                       
            bout=plt.scatter(bouts_xpos,bouts_ypos,c='red', alpha=0.5)
            ax.legend([path,obj,bout],['path','object','bout'],loc='best')
            ax.set_yticks([]) #removes y ticks from plot
            ax.set_xticks([]) #removes x ticks from plot
            ax.set_title('Example_fig: mouse_'+str(mouse_id))
            fig.show()
    
    bouts_data.to_csv(filespath+'\Data_bouts.csv', index_label='mouse_id') #write output to csv in filepaths dir 

    return bouts_data


filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master'

dat=object_bouts(filespath,40)
