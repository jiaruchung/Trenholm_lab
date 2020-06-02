# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:29:32 2020

@author: Kadjita Asumbisa and Jia-Ru Chung 
"""
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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
        occu_mat: occupancy matrix of given animal posX&Y
    """
    #Extracting x and y pos and bin lims
    xpos=animal_pos.iloc[:,0]
    ypos=animal_pos.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
    occu_mat, _, _ = np.histogram2d(xpos, ypos, [xbins,ybins]) #creating a 2d matrix of xpos and ypos defined by bins sizes on line36&37
    occu_mat= occu_mat[::-1] #flips the matrix to align with original path plot
    return occu_mat


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
    
    #Ask user to confirm if filespath is set to folder with processed (clean) files
    clean_files=input('Is filespath set to the clean files directory?: (y/n) ').lower()
    if clean_files=='y':
        
        print('realigning all object positions to upper left quadrant...')
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
    else:
        print('set filespath to folder with clean excel files before running this function')


filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master'

data=align_allMats(filespath, show_fig=True)

#collect the realigned occupancy matrices generated in for two conditions
cond1_filespath= r'C:\Users\kasum\Downloads\Trenholm_lab-master' #example path to condtion1  .xlsx files
cond1 = align_allMats(cond1_filespath, show_fig=False)

cond2_filespath=r'C:\Users\kasum\Downloads\Trenholm_lab-master' #example path to condtion2  .xlsx files
cond2 = align_allMats(cond2_filespath, show_fig=False)




def collect_samples(aligned_data):
    """
    Args: 
        urs: combinations of heatmaps aligned to the upper-left corner resulted by the get_urs function
    
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
            
    #Populating line 134 with the extracted value pairs for each occupancy bin with data from all animals
    for occupancy in enumerate(aligned_data):
        for i in range(bins):
            for j in range(bins):
                cells_val_pair[i][j].append(occupancy[1][i][j])
    return cells_val_pair

cond1_samples = collect_samples(cond1)
cond2_samples = collect_samples(cond2)    



# generate a list for the rank_sum test results between two conditions in each bin
from scipy.stats import ranksums
import seaborn as sns
 
ranksum_test_on_cells = []
bins=10
for i in range(bins):
    ranksum_test_on_cells.append([])
    for j in range(bins):
        
        # applying the rank_sum test to animal's occupancy between 2 conditions by looking at cell samples
        ranksum_test_on_cells[i].append(ranksums(cond1_samples[i][j], cond2_samples[i][j]))
      
result = list(map(lambda row: list(map(lambda element: element.pvalue, row)), ranksum_test_on_cells))
  
####### p_value HeatMap Plot for 2 Conditions##########    
print('Displaying difference of two samples')

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

#Notes
#put experimental and control data in seperate directories
#if we want all to be in the same fucntion, then we need to set another loop

#improve code to cater for irregularly shaped environments such as those with weird bin shapes
#as this code stands now, you have to set directory to control and experimental gorups but it will be more 
#efficient if theya re all in the same location but the code uses file names to seperate and prcoces them 
#from same dir.