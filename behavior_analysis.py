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

import numpy as np
import pandas as pd
from pylab import *
import os, sys
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter
#from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
sns.set(style="ticks", color_codes=True)

###############################################################################
import os
import glob

#plotting automation 

files=glob.glob(os.path.join(r'F:\training_videos\m vs c','*.xlsx'))

for idx,file in enumerate(files):
    pos=pd.read_excel(file)
    dx = array(pos.X[1:])-array(pos.X[:-1]); dx=np.concatenate(([0],dx))
    dy = array(pos['Y'][1:])-array(pos['Y'][:-1]); dy=np.concatenate(([0],dy))
    
    for i,x in enumerate(dx):
        if abs(dx[i])>50: #Tracking noise threshold
            pos['X'][i]= pos['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
            pos['Y'][i]=NaN
    
#    figure()
#    plot(pos['X'],pos['Y'])
#    xmin=min(pos['X']); xmax=max(pos['X'])
#    ymin=min(pos['Y']); ymax=max(pos['Y'])
#    plt.gca().invert_yaxis
    fig,az=plt.subplots()
    plot(pos['X'],pos['Y'])
    xmin=min(pos['X']); xmax=max(pos['X'])
    ymin=min(pos['Y']); ymax=max(pos['Y'])
    az.invert_yaxis()   

    xaxis=plot([xmin, xmin],[ymin, ymax],color='k')
    yaxis=plot([xmin, xmax],[ymax, ymax],color='k')
    xaxis1=plot([xmax, xmax],[ymin, ymax],color='k')
    yaxis1=plot([xmin, xmax],[ymin, ymin],color='k')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.axis('off')
    plt.show
    plt.savefig(file+'.png',figsize=(5,5),dpi=600)
    
#xaxis=plot([(xmin+xmax)/2,(xmin+xmax)/2],[ymin,ymax],color='k')
#yaxis=plot([xmin,xmax],[(ymax+ymin)/2,(ymax+ymin)/2],color='k')
#plt.rcParams['axes.spines.right'] = False
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.left'] = False
#plt.rcParams['axes.spines.bottom'] = False
#plt.title('M10-c2') 


#looping for cleaning data and generating heatmap
import os
import glob

files=glob.glob(os.path.join(r'F:\training_videos\c vs m\Males-c','*.xlsx'))
for file in files:
    pos=pd.read_excel(file)
    dx = array(pos.X[1:])-array(pos.X[:-1]); dx=np.concatenate(([0],dx))
    dy = array(pos['Y'][1:])-array(pos['Y'][:-1]); dy=np.concatenate(([0],dy))
    
    for i,x in enumerate(dx):
        if abs(dx[i])>50: #Tracking noise threshold
            pos['X'][i]= pos['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
            pos['Y'][i]=NaN
    pos.to_excel(file)


#for i in range(2):
#    pass
#figure()
#ax=subplot(20,40,1+i)

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

def heatmap_mat(occu):
    bins=10 
    xpos=pos['X']
    ypos=pos['Y']
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
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
    return q

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



#figure(); occu_heatmap(pos)
#flip the heatmap ligned with same corners
    
files=glob.glob(os.path.join(r'F:\training_videos\c vs ddm\Males-c','*.xlsx'))
for idx, file in enumerate(files):
    subplot(4,5,idx+1)
    pos=pd.read_excel(file)
    a,b=occu_heatmap(pos)

urs=[]
for idx,file in enumerate(files):
    name=files[idx].split('-')[-1].split('.')[-2]
    pos=pd.read_excel(file)
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

vals=0
for i in range(len(urs)):
#    vals+=urs[i]
    vals=vals+urs[i]
    
figure(); heatmap_mat(vals)
#plt.savefig('heatmaps/Blinds-c +'.eps',figsize=(5,5),dpi=600) 
            
plt.savefig('heatmaps/reverse-c.eps',figsize=(5,5), dpi=600)
    
#clean_pos=pd.DataFrame(index=np.arange(len(pos)),columns=['X','Y'])
#for i,x in enumerate(dx):
#    if abs(dx[i])>25: #Tracking noise threshold
#        clean_pos.iloc[i,0]=pos['X'][i-1]# NaN#data['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
#        clean_pos.iloc[i,1]=pos['Y'][i-1]
#    else:
#        clean_pos.iloc[i,0]=pos['X'][i]
#        clean_pos.iloc[i,1]=pos['Y'][i]
#fig,az=plt.subplots()
#plot(clean_pos['X'],clean_pos['Y'])
#figure();plot(pos['X'],pos['Y'])

#out-of-scale plot
idx=pos['X']<475
idx2=pos['Y'] >46
#xmin=min(idx); xmax=max(idx)
#ymin=min(idx2); ymax=max(idx2)
pos_new=pos[idx]
pos_new=pos_new[idx2]
fig,az=plt.subplots()
plot(pos_new['X'],pos_new['Y'])
xmin=min(pos_new['X']); xmax=max(pos_new['X'])
ymin=min(pos_new['Y']); ymax=max(pos_new['Y'])
az.invert_yaxis()
#plt.gca().set_aspe
xaxis=plot([xmin, xmin],[ymin, ymax],color='k')
yaxis=plot([xmin, xmax],[ymax, ymax],color='k')
xaxis1=plot([xmax, xmax],[ymin, ymax],color='k')
yaxis1=plot([xmin, xmax],[ymin, ymin],color='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.axis('off')
plt.savefig('M13-m1.eps',dpi=300)



#############################################################################
#Hist
############################################################################
val=np.arange(min(dx), max(dx) + 1, 1)
figure();gca().set_xlim(0,100)
hist(abs(dx),bins=val)

############################################################################
#single heatmap

pos=pd.read_excel(r'C:\Users\labuser\Desktop\Partha Dark_Light Room\ka.xlsx')
figure();plot(pos['X'],pos['Y'])

def occu_heatmap(pos):  
    bins=10 
    xpos=pos['X']
    ypos=pos['Y']
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, bins+10)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, bins+1)
    occu, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    occu=gaussian_filter(occu,sigma=0.7)
    #fig,ax=plt.subplots()
    q = imshow(occu, cmap='jet', interpolation='bilinear')
    #ax.axis('off')
    #cbar=fig.colorbar(q,orientation='vertical')
    #cticks=cbar.ax.get_xticks()
    #cbar.set_ticks([])
    gca().invert_yaxis()
    return q,occu
figure(); occu_heatmap(pos)


##############################################################################

#calculation #single aniaml

pos=pd.read_excel(r'F:\training_videos\031020\rM10-c.xlsx')#CHANGE THIS PATH ONLY
#figure();plot(pos['X'],pos['Y'])

#Calculate distance between consecutive x,y points
dx = array(pos['X'][1:])-array(pos['X'][:-1]); dx=np.concatenate(([0],dx))
dy = array(pos['Y'][1:])-array(pos['Y'][:-1]); dy=np.concatenate(([0],dy))


for i,x in enumerate(dx):
    if abs(dx[i])>50: #Tracking noise threshold
        pos['X'][i]=pos['X'][i-1]# NaN#data['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
        pos['Y'][i]=pos['Y'][i-1]
        
fig,az=plt.subplots()
plot(pos['X'],pos['Y'])
xmin=min(pos['X']); xmax=max(pos['X'])
ymin=min(pos['Y']); ymax=max(pos['Y'])
az.invert_yaxis()
xaxis=plot([(xmin+xmax)/2,(xmin+xmax)/2],[ymin,ymax],color='k')
yaxis=plot([xmin,xmax],[(ymax+ymin)/2,(ymax+ymin)/2],color='k')


pos_x=array(pos['X'])
pos_y=array(pos['Y'])
fr= 30 #camera frame rate

#you may have to manually define the center of your environment for data with irregular path plots
x_cen= (pos_x.max()+pos_x.min())/2 
y_cen=(pos_y.max()+pos_y.min())/2
c_vert=plot([x_cen,x_cen], [pos_y.min(),pos_y.max()]) #plots vertical line through center for verification
c_hor=plot([pos_x.min(),pos_x.max()],[y_cen,y_cen])   #plots horizontal line through center for verification


#DISTANCE TRAVELLED
dx = pos_x[1:]-pos_x[:-1]
dy = pos_y[1:]-pos_y[:-1]
dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points

#Quadrant
#upper left
up_left=(x_cen>pos_x) & (y_cen>pos_y)
up_left_allDis=dist[up_left] #Extracting distance covered in quadrant
up_left_totDis=sum(up_left_allDis) #distance covered
up_left_totTime=len(up_left_allDis)/fr  #total time in seconds
up_left_vel=up_left_totDis/up_left_totTime  #velocity in quadrant
print('Dist in upper-left quadrant = ', up_left_totDis)
print('Durations in upper-left quadrant = ', up_left_totTime)

#upper right
up_right=(x_cen<pos_x) & (y_cen>pos_y)
up_right_allDis=dist[up_right] #Extracting distance covered in quadrant
up_right_totDis=sum(up_right_allDis) #distance covered
up_right_totTime=len(up_right_allDis)/fr  #total time in seconds
up_right_vel=up_right_totDis/up_right_totTime  #velocity in quadrant
print('Dist in upper-right quadrant = ', up_right_totDis)
print('Durations in upper-right quadrant = ', up_right_totTime)

#buttom left
bo_left=(x_cen>pos_x) & (y_cen<pos_y) #defining quadrant
bo_left_allDis=dist[bo_left] #Extracting distance covered in quadrant
bo_left_totDis=sum(bo_left_allDis) #distance covered
bo_left_totTime=len(bo_left_allDis)/fr  #total time in seconds
bo_left_vel=bo_left_totDis/bo_left_totTime  #velocity in quadrant
print('Dist in bottom-left quadrant = ', bo_left_totDis)
print('Durations in bottom-left quadrant = ', bo_left_totTime)

#lower right
bo_right=(x_cen<pos_x) & (y_cen<pos_y)
bo_right_allDis=dist[bo_right] #Extracting distance covered in quadrant
bo_right_totDis=sum(bo_right_allDis) #distance covered
bo_right_totTime=len(bo_right_allDis)/fr  #total time in seconds
bo_right_vel=bo_right_totDis/bo_right_totTime  #velocity in quadrant
print('Dist in bottom-right quadrant = ', bo_right_totDis)
print('Durations in bottom-right quadrant = ', bo_right_totTime)
#----------------------------------------------------------------------------
#tot dist 
tot_dist=sum([up_left_totDis,up_right_totDis,bo_left_totDis,bo_right_totDis])
print('Total distance = ', tot_dist)

#subzones for objects
x_top_left_object_zone= plot([xmin,xmin+75],[ymax-75,ymax-75],color='k')  
y_top_left_object_zone= plot([xmin+75,xmin+75],[ymax,ymax-75],color='k')

x_top_right_object_zone= plot([xmax-75,xmax],[ymax-75,ymax-75],color='k')
y_top_right_object_zone= plot([xmax-75,xmax-75],[ymax,ymax-75],color='k')  

x_bottom_left_object_zone= plot([xmin,xmin+75],[ymin+75,ymin+75],color='k') 
y_bottom_left_object_zone= plot([xmin+75,xmin+75],[ymin,ymin+75],color='k')

x_bottom_right_object_zone= plot([xmax-75,xmax],[ymin+75,ymin+75],color='k') 
y_bottom_right_object_zone= plot([xmax-75,xmax-75],[ymin,ymin+75],color='k')


# object approaching attempt bouts and duration time 

#upper left
up_object_left= (xmin+75>pos_x) & (pos_y<ymin+75)
up_object_left_allDis=dist[up_object_left]
up_object_left_totDis=sum(up_object_left_allDis) 
up_object_left_totTime=len(up_object_left_allDis)/fr  #total time in seconds
print('Dist in upper-left subzone = ', up_object_left_totDis)
print('Durations in upper-left subzone = ', up_object_left_totTime)

#upper right
up_object_right= (xmax-75<pos_x) & (pos_y<ymin+75)
up_object_right_allDis=dist[up_object_right]
up_object_right_totDis=sum(up_object_right_allDis) 
up_object_right_totTime=len(up_object_right_allDis)/fr  #total time in seconds
print('Dist in upper-right subzone = ', up_object_right_totDis)
print('Durations in upper-right subzone = ', up_object_right_totTime)

#bottom left
bo_object_left= (xmin+75>pos_x) & (pos_y>ymax-75)
bo_object_left_allDis=dist[bo_object_left] #Extracting distance covered in subzone
bo_object_left_totDis=sum(bo_object_left_allDis) #distance covered
bo_object_left_totTime=len(bo_object_left_allDis)/fr  #total time in seconds
print('Dist in bottom-left subzone = ', bo_object_left_totDis)
print('Durations in bottom-left subzone = ', bo_object_left_totTime)

#bottom right      
bo_object_right= (xmax-75<pos_x) & (pos_y>ymax-75)
bo_object_right_allDis=dist[bo_object_right] #Extracting distance covered in subzone
bo_object_right_totDis=sum(bo_object_right_allDis) #distance covered
bo_object_right_totTime=len(bo_object_right_allDis)/fr  #total time in seconds
print('Dist in bottom-right subzone = ', bo_object_right_totDis)
print('Durations in bottom-right subzone = ', bo_object_right_totTime)




plot(xmax-75<pos_x, pos_y<ymin+75)

###############################################################################
#PLOTS
###############################################################################
figure(); plot(pos_x,pos_y)  #Full covereage

quad=up_left #define quadrant e.g up_right, up_left, b_left

p_x=pos_x[quad]
p_y=pos_y[quad] 
plot(p_x,p_y) # plot defined quadrant





##############################################################################
#COMPRESSED CODE
##############################################################################
pos_x=array(pos_['X'])
pos_y=array(pos_['Y'])

fr= 120 #camera frame rate

x_cen=(pos_x.max()+pos_x.min())/2  
y_cen=(pos_y.max()+pos_y.min())/2

#DISTANCE TRAVELLED
dx = pos_x[1:]-pos_x[:-1]
dy = pos_y[1:]-pos_y[:-1]
dist = np.concatenate(([0],np.sqrt(dx**2+dy**2)))  #computes the distance between two consecuitive x,y points

#DEFINE ZONES
up_left=(x_cen>pos_x) & (y_cen<pos_y)
up_right=(x_cen<pos_x) & (y_cen<pos_y)
b_left=(x_cen>pos_x) & (y_cen>pos_y)
b_right=(x_cen<pos_x) & (y_cen>pos_y)

quads=[up_left,up_right, b_left, b_right]

data=pd.DataFrame(index=(['up_left', 'up_right', 'b_left', 'b_right']), columns=(['tot_dist','tot_time','vel']))
for i,x in enumerate (range(len(quads))):
    allDis=dist[quads[i]] #Extracting distance covered in quadrant
    totDis=sum(allDis) #distance covered   MOdify this to reflect the appropriate units
    totTime=len(allDis)/fr  #total time in seconds
    vel=totDis/totTime  #velocity in quadrant
    data.iloc[i,0]=totDis
    data.iloc[i,1]=totTime
    data.iloc[i,2]=vel
    
    
 
##############################################################################
#plots and stats
##############################################################################
#day 1 plot
time_data = pd.read_csv('Total time spent_stats_blind.csv')
dat=time_data[['computer','mouse']]
means=pd.DataFrame(index=dat.columns, columns=np.arange(1))
means.loc['computer',0]= dat['computer'].mean()
means.loc['mouse',0]=dat['mouse'].mean()
fig=figure()
plot(dat.T, color='grey'); plot(means,color='black', linewidth=3)
#plt.title('More time spent with 3D computer object than mouse object') 
#plt.xlabel('Categories') 
plt.ylabel('Time spent with object (s) ') 
#plot([0,1],[150,150], color = 'k')
plt.text(0.45, 150, 'P=0.17', verticalalignment='center')

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('blind.eps',dpi=300)


#day 2 plot 
time_data = pd.read_csv(r'Total time spent_stats_sighted_ddm_control.csv')
dat=time_data[['computer','DDM']]
means=pd.DataFrame(index=dat.columns, columns=np.arange(1))
means.loc['computer',0]= dat['computer'].mean()
means.loc['DDM',0]=dat['DDM'].mean()
fig=figure()
plot(dat.T, color='grey'); plot(means,color='black', linewidth=3)
#plt.title('More time spent with 3D computer object than mouse object') 
#plt.xlabel('Categories') 
plt.ylabel('Time spent with object (s) ') 
#plot([0,1],[150,150], color = 'k')
plt.text(0.45, 200, 'P=0.39', verticalalignment='center')

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.savefig('sighted_ddm_control.eps',dpi=300)

#stats
time_data = pd.read_csv('Total time spent_stats_sighted_ddm_control.csv')
from numpy import ndarray
computer=time_data['computer']
mouse=time_data['DDM']
computer_mean = time_data['computer'].mean()
mouse_mean = time_data['DDM'].mean()
import scipy.stats
scipy.stats.wilcoxon(computer, mouse)

#stats day 2
time_data = pd.read_csv('Total time spent_stats_males.csv')
from numpy import ndarray
computer=time_data['computer day 2']
mouse=time_data['mouse day 2']
computer_mean = time_data['computer day 2'].mean()
mouse_mean = time_data['mouse day 2'].mean()
import scipy.stats
scipy.stats.wilcoxon(computer, mouse)

#rank plot
time_data = pd.read_csv('Total time spent_stats_males.csv')
time_data.describe()
time_data.head()
sns.set(font_scale=2) 
x = np.random.normal(size=37)
y = np.random.lognormal(size=37)
fig, ax = plt.subplots()
ax = sns.catplot(x='ranking',y='ratio first day', marker='s', color='k', label='big', size =8, alpha=0.5, linewidth=5, aspect=0.7,
                 data=time_data)
ax.fig.autofmt_xdate()
xlabel('Ranking', fontsize=20)
ylabel('Ratio of time spend in the computer over mouse quadrant', fontsize=20)

#ratio plot 
time_data = pd.read_csv('Total time spent_stats.csv')
time_data.describe()
time_data.head()
ax = sns.catplot(x='ratio first day',y='ratio second day',hue='Mice ID',kind='box',data=time_data)
ax.fig.autofmt_xdate()



#lineplot 
x = [2, 4, 6]
y = [1, 3, 5]
plt.plot(x, y)
plt.show()

#Number of attacks
plt.subplot(2,1,2)
Time_spent_ratio=dat['Time spent ratio']  #the larger the more pronouced effect of object avoidance 
Number_of_attacks=dat['Number of attacks']     
plt.scatter(Number_of_attacks, Time_spent_ratio)
plt.title('Correlation between social defeat and object avoidance') 
plt.xlabel('Number of attacks') 
plt.ylabel('Ratio of total time spent') 

#Fighting durations 

plt.subplot(2,1,2)
Time_spent_ratio=dat['Time spent ratio']  #the larger the more pronouced effect of object avoidance 
Fighting_durations=dat['Fighting durations']    
plt.scatter(Fighting_durations, Time_spent_ratio)
plt.title('Correlation between social defeat and object avoidance') 
plt.xlabel('Fighting durations in secs')
plt.ylabel('Ratio of total time spent') 

