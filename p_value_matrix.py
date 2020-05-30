# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:29:32 2020

@author: Kadjita Asumbisa and Jia-Ru Chung 
"""
import numpy as np
import pandas as pd
from pylab import *
import os
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter
from scipy.stats import ranksums
import seaborn as sns
import glob


<<<<<<< Updated upstream
#flipping functions of the heatmaps 
=======
###############################################################################



#automatically generating heatmap
for idx, file in enumerate(filename_list):

        pos=pd.read_excel(file)

    name=filename_list[idx].split('-')[-1].split('.')[-2]
        if name=='ur':
            _,mur=occu_heatmap(pos)
            flipped_from_ur=ur(mur)
            urs.append(flipped_from_ur) 
        elif name=='bl':
            _,mbl=occu_heatmap(pos)
            flipped_from_bl=bl(mbl)
            urs.append(flipped_from_bl)
        elif name=='br':
            _,mbr=occu_heatmap(pos)
            flipped_from_br=br(mbr)
            urs.append(flipped_from_br)
        elif name=='ul':
            _,mul=occu_heatmap(pos)
            static=ul(mul)
















>>>>>>> Stashed changes
def ur(m): 
    first=np.flip(m,axis=1)
    return first       
def bl(m): 
    first=np.flip(m,axis=0)
    return first         
def br(m):
    first=np.flip(m,axis=0)
    sec=np.flip(first,axis=1)
    return sec  
def ul(m):
    first = m
    return first 




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

def get_urs(filename_list):
    urs=[]
    print(filename_list)
    for idx, file in enumerate(filename_list):

        pos=pd.read_excel(file)

        dx = array(pos.X[1:])-array(pos.X[:-1])
        dx = np.concatenate(([0],dx))
        # dy = array(pos['Y'][1:])-array(pos['Y'][:-1])
        # dy = np.concatenate(([0],dy))

        for i,x in enumerate(dx):
            if abs(dx[i])>50: #Tracking noise threshold
                pos.loc[[i], ['X']] = pos.loc[[i-1], ['X']]
                # pos['X'][i]= pos['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
                # pos['Y'][i]=NaN

        name=filename_list[idx].split('-')[-1].split('.')[-2]
        if name=='ur':
            _,mur=occu_heatmap(pos)
            flipped_from_ur=ur(mur)
            urs.append(flipped_from_ur) 
        elif name=='bl':
            _,mbl=occu_heatmap(pos)
            flipped_from_bl=bl(mbl)
            urs.append(flipped_from_bl)
        elif name=='br':
            _,mbr=occu_heatmap(pos)
            flipped_from_br=br(mbr)
            urs.append(flipped_from_br)
        elif name=='ul':
            _,mul=occu_heatmap(pos)
            static=ul(mul)
            urs.append(static)
    return urs

#Paths to all the data files in the folder of condition 1 and 2 
condition_1_filenames = glob.glob(os.path.join('./Blinds-c-excel','*.xlsx'))
condition_2_filenames = glob.glob(os.path.join('./Blinds-m-excel','*.xlsx'))

# collect the heatmaps in both conditions 
urs1 = get_urs(condition_1_filenames)
urs2 = get_urs(condition_2_filenames)

#number of bin in the arena 
BINS = 10 

def collect_samples(urs):
    
    """
    Args: 
        urs: combinations of heatmaps aligned to the upper-left corner resulted by the get_urs function 
    
    Returns: 
        result: a list of results rank sum test between two conditions in each bin 
        pval_heatmap2: the heatmap showing p-value in each bin as a result of rank_sum test between two conditions
        
    """
    cells = []
    for i in range(BINS):
        cells.append([])
        for j in range(BINS):
            cells[i].append([])

    for occu in enumerate(urs):
        for i in range(BINS):
            for j in range(BINS):
                cells[i][j].append(occu[1][i][j])
    return cells

urs1_samples_by_cell = collect_samples(urs1)
urs2_samples_by_cell = collect_samples(urs2)              

ranksum_test_on_cells = []

for i in range(BINS):
    ranksum_test_on_cells.append([])
    for j in range(BINS):
        ranksum_test_on_cells[i].append(ranksums(urs1_samples_by_cell[i][j], urs2_samples_by_cell[i][j]))

result = list(map(lambda row: list(map(lambda element: element.pvalue,
                                       row)),
                  ranksum_test_on_cells))
  
    
print('Displaying difference of two samples')


#heatmap of the result 
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







