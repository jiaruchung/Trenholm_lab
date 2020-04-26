import numpy as np
import pandas as pd
from pylab import *
import os
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter
from scipy.stats import ranksums
import seaborn as sns
import glob

# from matplotlib_scalebar.scalebar import ScaleBar

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:21:21 2019

@author: labuser
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:29:32 2019

@author: kasum
"""

sns.set(style="ticks", color_codes=True)

###############################################################################



#automatically generating heatmap
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
    bins=10 
    xpos=pos['X']
    ypos=pos['Y']
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
    occu, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    occu=gaussian_filter(occu,sigma=0.7)
    fig,ax=plt.subplots()
    q = imshow(occu, cmap='jet', interpolation='bilinear')
    gca().set_yticks([])
    gca().set_xticks([])
    #ax.axis('off')
    cbar=fig.colorbar(q,orientation='vertical')
    cticks=cbar.ax.get_xticks()
    cbar.set_ticks([])
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

# vals=0
# for i in range(len(urs)):
# #    vals+=urs[i]
#     vals=vals+urs[i]
            
# plt.savefig('heatmaps/reverse-c.eps',figsize=(5,5), dpi=600)

condition_1_filenames = glob.glob(os.path.join('./Males-c-excel','*.xlsx'))
condition_2_filenames = glob.glob(os.path.join('./Males-m-excel','*.xlsx'))

urs1 = get_urs(condition_1_filenames)
urs2 = get_urs(condition_2_filenames)

print(urs1)
print(urs2)
# vals2=0
# for i in range(len(urs2)):
# #    vals+=urs[i]
#     vals2=vals2+urs2[i]
    
#rank_sum test for each value in 2 conditions 

# ranksums(flipped_from_ur[:][:], flipped_from_ur2[:][:])
# ranksums(static[:][:], static2[:][:])
# ranksums(flipped_from_br[:][:], flipped_from_br2[:][:])
# ranksums(flipped_from_bl[:][:], flipped_from_bl2[:][:])
