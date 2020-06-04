# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:52:21 2020

@author: Kadjita Asumbisa & Jia-Ru Chung
"""
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from scipy.stats import ranksums



def get_files_info(filespath):
    """
    Args: filepaths: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2)
    
    Returns:
        files: paths to all .xlsx files in filespath dir
        ids: index of each animal_id stored as a list  
    """
    files=glob.glob(os.path.join(filespath,'*.xlsx')) #storing the dir of all .xlsx files input dir
    ids=[int(file.split('\\')[-1].split('_')[0]) for idx,file in enumerate(files)] #Extracting animal_id's
    
    return files,ids


def clean_pos_data(filespath,thres=.90):
    """
    Args: raw_filespath: directory containing .xlsx raw data files
        thres: threshold for dermining the distribution of data to keep during cleaning. Ideally thus must be manually adjusted for each plot
        
    Outputs:
        figs: showing the raw and processed path plots for each animal 
        animal_pos: processed (clean) .xlsx version of each file saved to clean_filespath dir with original filename      
    """
    new_directory='\cleanData'
    #Creates a the new directory if it doesnt exist, else the code passes and executes the remaining lines
    try:
        os.mkdir(filespath+new_directory) #Create a new directory to store processed files
        print('Creating new directory to write processed files to...')
    except:
        print('Directory for writing processed files already exist')
        pass

    files,_=get_files_info(filespath) #function call to return only first return variable
    
    for idx, file in enumerate(files):
        file_info=str('\\')+ file.split('\\')[-1]
        position_data=pd.read_excel(file) #read_excel file with position data
        animal_pos=position_data.loc[:,('X','Y')]
        
        #computing change in x and y
        dx = np.array(animal_pos.iloc[:,0][1:])-np.array(animal_pos.iloc[:,0][:-1]); 
        dy = np.array(animal_pos.iloc[:,1][1:])-np.array(animal_pos.iloc[:,1][:-1]); 
    
        #### Raw Data Plot###########
        plt.figure()
        plt.suptitle('Mouse_'+file.split('\\')[-1].split('_')[0]+' Path Plots')
        ax=plt.subplot(211) 
        ax.set_title('Raw_data')      
        ax.plot(animal_pos.iloc[:,0],animal_pos.iloc[:,1])
        ax.set_xticks([])
        ax.set_yticks([])
           
        hist_counts,hist_bins=np.histogram(abs(dx)) #generates a histogram counts and boundaries(bins) of delta x (change in x pos per frame)
        counts=0
        jumps_in_tracking=False #Assumes the tracking has no jumps
        
        #Identifying the hist_bin edge which contains <= thres(i.e 95%) of the cumulative hist counts
        for i in range(len(hist_counts)):
            if counts <= len(dx)*thres:
                counts+=hist_counts[i] #cumulate counts from hist_counts
                
                #bins beyond the threshold bins are considered noisy dx
                threshold=hist_bins[i+1] #+1 here ensures that the bin edge is captured once the bin_counts pos is identified
                jumps_in_tracking=True
        
        #if jumps in tracking, enter this loop        
        if jumps_in_tracking:
            for i in range(len(dx)):
                if abs(dx[i])>threshold: #Tracking noise threshold
                    animal_pos.loc[i]= np.NaN #replacing positional jumps with NaN in original position data
        
        #Since we are using distance to determine cut offs, we can use any axis and extend to the other
                if abs(dy[i])>threshold: #Here we are using threshold determined from dx on dy to set them to NaNs in original positio data
                    animal_pos.loc[i]= np.NaN
        else:
            continue
     
        #### Processed Data plot##############        
        ax2=plt.subplot(212)       
        ax2.plot(animal_pos.iloc[:,0],animal_pos.iloc[:,1])
        ax2.set_title('Processed_data')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        animal_pos.dropna().reset_index(drop=True)# resetting may be redundant bcos of next line check on it 
        print('writing processed file '+ file_info +' to cleanData directory...')
        animal_pos.to_excel(filespath+new_directory+file_info, index=False) #save file to new directory


         
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
        Saved Output in a pandas dataframe  
    """    
    filespath=filespath+'\cleanData' #change directory to clean folder
    
    files,_=get_files_info(filespath) #calls external fxn to get files info

    #creating pandas dataframe to hold data
    quad_coverage=pd.DataFrame(columns=['cond','sex','object_loc','ul_dist', 'ul_dur',\
                       'ul_vel','ur_dist', 'ur_dur','ur_vel', \
                              'bl_dist', 'bl_dur','bl_vel','br_dist', 'br_dur','br_vel']) 
    
    for idx,file in enumerate(files): 
        mouse_id=file.split('\\')[-1].split('_')[0]
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
            #Logical indexing for identifying differemt quadrants based on relative loc referenced from center on environment
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
            
        quad_coverage.loc[mouse_id,(quad_coverage.columns[3:])]= computed_vals #assigning computed_vals to current mouse_id and respective cols in dataframe on line44
        
    quad_coverage.to_csv(filespath+'\Data_quadrants.csv', index_label='mouse_id') #write data to filespath as csv
    
    return quad_coverage



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

        #Assigning extracted data to be stored in their respective locs in the dataframe   
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


def occu_matrix(animal_pos,bins=10):
    """
    Args: 
        animal_pos: pandas dataframe animal pos X and Y in cols 1 and 2 in the folder of two conditions 
        bins: bin size for determining size of matrix. Default set to 10
    Returns: 
        occu_mat: occupancy matrix of given animal posX&Y
    """
    #Extracting x and y pos and bin lims
    xpos=animal_pos.iloc[:,0]
    ypos=animal_pos.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
    occu_mat, _, _ = np.histogram2d(xpos, ypos, [xbins,ybins]) #creating a 2d matrix of xpos and ypos defined by bins sizes
    occu_mat= occu_mat[::-1] #flips the matrix to align with original path plot
    return occu_mat


def occu_plots(filespath):
    """
    Args: 
        filespath: Path to folder containing all animals pos .xlsx data files ( X & Y corresponding to cols 1&2)
    
    Outputs & Returns: 
        fig: showing trajectories of all animals in a fig subplots
        fig1: showing heatmaps of the trajectory plots
    """
    filespath=filespath+'\cleanData' #change directory to clean folder

    files,ids=get_files_info(filespath)
    
    #PLOT1: Trajectory line plot
    fig,ax=plt.subplots()
    rows=len(files)//2
    for i,file in enumerate(files):
        mouse_info=file.split('\\')[-1].split('.')[0] #splits and assigns needed info to var
        position=pd.read_excel(file) #reading position file into a pandas dataframe
        animal_pos=position.iloc[:,:2] #extracting posx and y which corresponds to cols 1&2
        xpos=animal_pos.iloc[:,0]
        ypos=animal_pos.iloc[:,1]
        
        plt.subplot(rows,(len(files)//2)+2,i+1) 
        plt.plot(xpos,ypos)
        plt.gca().set_yticks([])
        plt.gca().set_xticks([])
        plt.title('Mouse_info: '+mouse_info) 
        fig.show()
        
        
    #PLOT2: Trajecory heatmap plot
    fig1,ax1=plt.subplots()
    for i,file in enumerate(files):
        mouse_info=file.split('\\')[-1].split('.')[0] 
        plt.subplot(rows,(len(files)//2)+2,i+1)
        position=pd.read_excel(file) 
        animal_pos=position.iloc[:,:2] 
        occu=occu_matrix(animal_pos)  #fxn call to generate occupancy matrix
        occu=gaussian_filter(occu,sigma=0.7)  #using a guassian filter to smoothen occupancy matrix
        
        plt.title('Mouse_info: '+mouse_info)
        heatmap=plt.imshow(occu, cmap='jet', interpolation='bilinear') 
    
        plt.gca().set_yticks([])
        plt.gca().set_xticks([])
        cbar=fig1.colorbar(heatmap,orientation='vertical') #color bar legend
        cbar.ax.get_xticks()
        cbar.set_ticks([])
        cbar.ax.set_ylabel('Occupancy')
        min_=plt.text(11,9.5,'min')
        max_=plt.text(11,-0.3,'max')
        fig1.show()

    return fig,fig1


def align_allMats(filespath, show_fig=True):
    """
    Args: 
        filespath: Path to all the .xlsx files with animal pos X and Y in cols 1 and 2 in the folder of two conditions 
        show_fig (boolean): shows fig if set to True. Default is set to True
    Outputs:
            fig: a heatmap of of all combined pos matrices if show_fig is set to True
    Returns: 
        alingned_to_ul: occupancy matrix of all animals in dir where object loc aligned to the upper-left quadrant 
    """ 
    
    filespath=filespath+'\cleanData' #change directory to clean folder
    
    files,_=get_files_info(filespath) #function call to return all file directories
    
    aligned_to_ul=[] #Initializing an empty list to hold realigned position matrix 
    for file in files:
        position=pd.read_excel(file) #reading position file into a pandas dataframe
        animal_pos=position.iloc[:,:2] #extracting posx and y which corresponds to cols 1&2

        object_loc=file.split('\\')[-1].split('_')[2] #Extracting object location from string
        
        #flip position matrix depending on object loc to ensure that all objects locs across animals are aligned to upper left quadrant 
        if object_loc=='ur': 
            ur_matrix=occu_matrix(animal_pos) #fxn call to compute occupancy matrix 
            flipped_from_ur=np.flip(ur_matrix,axis=1) #flip on second axis
            aligned_to_ul.append(flipped_from_ur) 
        elif object_loc=='bl':
            bl_matrix=occu_matrix(animal_pos)
            flipped_from_bl=np.flip(bl_matrix,axis=0)
            aligned_to_ul.append(flipped_from_bl)
        elif object_loc=='br':
            br_matrix=occu_matrix(animal_pos)
            flipped_from_br=np.flip(br_matrix,axis=0) #flip on first axis
            flipped_from_br=np.flip(flipped_from_br,axis=1) #flip on second axis
            aligned_to_ul.append(flipped_from_br) #append the after second flip
        else:
            ref_matrix=occu_matrix(animal_pos) #ul: arbitrarily pre-determined as reference object location for realignment 
            aligned_to_ul.append(ref_matrix)
            
    #### ALIGNED HEATMAP PLOT #########
    if show_fig:
        #initializing matrix of zeros with the shape of the dimensions of a single position matrix to hold the sum of all matrices across animals
        data=np.zeros(np.shape(aligned_to_ul)[1:3])
        for i in range(len(aligned_to_ul)):
            data+=aligned_to_ul[i]     #adds corresponding cells for all pos matrix across animals
        data=gaussian_filter(data,sigma=0.7)  #using a guassian filter to smoothen occupancy matrix
        fig,ax=plt.subplots()
        plt.title('Object Aligned to Upper Left Quadrant (N='+str(len(aligned_to_ul))+')') # the str attachement extracts the number of animals in the list
        heatmap=plt.imshow(data, cmap='jet', interpolation='bilinear') 
        ax.set_yticks([])
        ax.set_xticks([])
        cbar=fig.colorbar(heatmap,orientation='vertical') #color bar legend
        cbar.ax.get_xticks()
        cbar.set_ticks([])
        cbar.ax.set_ylabel('Occupancy')
        min_=plt.text(11,9.5,'min')
        max_=plt.text(11,-0.3,'max')
        fig.show()
       
    return aligned_to_ul
    
    
def collect_samples(aligned_data):
    """
    Args: 
        aligned_data: combinations of occupancy matrix aligned to the upper-left corner resulted from align_allMats fxn
    
    Returns: 
        cells: the resulting big matrix of each bin value pairs for all animals
    """
    cells_val_pair = []
    bins=np.shape(aligned_data)[-1]
    
    #Creating empty lists with desired dimensions to hold cell value pairs
    for i in range(bins): #Note: this only works for bins n=m in w X n X m list where w,n and m refres to the dimensions
        cells_val_pair.append([])
        for j in range(bins):
            cells_val_pair[i].append([])
            
    #Populating cells_val_pair with the extracted value pairs for each occupancy bin with data from all animals
    for occupancy in enumerate(aligned_data):
        for i in range(bins):
            for j in range(bins):
                cells_val_pair[i][j].append(occupancy[1][i][j])
    return cells_val_pair


def ranksum_pval_matrix(cond1_samples,cond2_samples):
    """
    Args:
        cond1_samples: output of collect_samples fxn made of bin value pairs for all animals in one condition. i.e Control condition
        cond2_samples: similar to cond1_samples, but for another condition. i.e Experimental condition
    Output:
        fig: showing a heatmap matrix representing the pvals at different locations in the environment 
    Return:
        result: matrix of pvals from computed ranksum for cond1_samples vs cond2_samples            
    """
    bins=10
    ranksum_test_on_cells = []
    for i in range(bins):
        ranksum_test_on_cells.append([])
        for j in range(bins):
            
            # applying the rank_sum test to animal's occupancy between 2 conditions by looking at cell samples
            ranksum_test_on_cells[i].append(ranksums(cond1_samples[i][j], cond2_samples[i][j]))

    result = list(map(lambda row: list(map(lambda element: element.pvalue, row)), ranksum_test_on_cells))
    
    #heatmap showing p-value in each bin as a result of rank_sum test between two conditions
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True, reverse=True)
    sns.set(font_scale=3)
    fig, ax = plt.subplots(1)
    pval_heatmap2 = sns.heatmap(result, 
                               vmin=0.001, 
                               vmax=0.05, 
                               cmap=cmap,
                               ax=ax,
                               xticklabels = False, 
                               yticklabels = False,
                               annot_kws={"size": 14},
                               cbar_kws={'label': 'P-value for the Wilcoxon rank sum statistic'})
    cbar = pval_heatmap2.collections[0].colorbar
    cbar.set_ticks([0.05, 0.01, 0.001, 0])
    cbar.set_ticklabels(['>=0.05', '0.01', '<=0.001', '0'])
    cbar.ax.invert_yaxis()
    plt.show()
    return result
#################################################################################################  
######################### TEST DATA #############################################################
#################################################################################################
import sys
sys.exit()  #Allows you to run all functions before running test 

#cond1
filespath=r'C:\Users\kasum\Desktop\COMP598\COMP598_data_files\cond1(non-mouse)'
clean_pos_data(filespath, thres=.9)
quad_analysis(filespath)
occu_plots(filespath)
aligned_data_cond1=align_allMats(filespath, show_fig=True)
cond1_samples=collect_samples(aligned_data_cond1)

#object_bouts
obj_bouts_path=r'C:\Users\kasum\Desktop\COMP598\COMP598_data_files\object_boutsFile' #it is is the only file with the required data
object_bouts(obj_bouts_path,60)

#con2
filespath=r'C:\Users\kasum\Desktop\COMP598\COMP598_data_files\cond2(mouse)'
clean_pos_data(filespath, thres=1)
quad_analysis(filespath) 
occu_plots(filespath)
aligned_data_cond2=align_allMats(filespath, show_fig=True)
cond2_samples=collect_samples(aligned_data_cond2)


#cond1 vs cond2
result=ranksum_pval_matrix(cond1_samples,cond2_samples)
