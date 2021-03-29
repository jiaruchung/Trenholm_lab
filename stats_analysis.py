# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:09:08 2020

@author: labuser
"""
import numpy as np
import pandas as pd
from pylab import *
import os, sys
import matplotlib.pyplot as plt 
#plots and stats
##############################################################################


#day 1 plot
time_data = pd.read_csv(r"D:\Trenholm Lab\Edith\Behavior\Data\males\trim_to_3_mins\total_time_males3mins.csv")
dat=time_data[['con1','con2']]
means=pd.DataFrame(index=dat.columns, columns=np.arange(1))
means.loc['con1',0]= dat['con1'].mean()
means.loc['con2',0]= dat['con2'].mean()
fig=figure()
plot(dat.T, color='grey'); 
plot(means,color='black', linewidth=3)
#plt.title('More time spent with 3D computer object than mouse object') 
#plt.xlabel('Categories') 
plt.ylabel('Time spent with object (s) ') 
#plot([0,1],[150,150], color = 'k')
plt.text(0.45, 150, '', verticalalignment='center')

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('males3mins.eps',dpi=300)


#stats1
time_data2 = pd.read_csv(r"D:\Trenholm Lab\Edith\Behavior\Data\males\trim_to_3_mins\total_time_males3mins.csv")
from numpy import ndarray
con1=time_data2['con1']
con2=time_data2['con2']
con1_mean = time_data2['con1'].mean()
con2_mean = time_data2['con2'].mean()
import scipy.stats
x = scipy.stats.wilcoxon(con1, con2)
print(x)