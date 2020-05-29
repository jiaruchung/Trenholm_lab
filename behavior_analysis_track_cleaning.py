# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:45:53 2020

@author: labuser
"""
import numpy as np
import pandas as pd
#from pylab import*
import matplotlib.pyplot as plt 
import sys as os


filename=r'C:\Users\kasum\Downloads\Trenholm_lab-master\data.xlsx' #filepath
position_data=pd.read_excel(filename) #read_excel file with position data
animal_pos=position_data.iloc[:,:2]

data=animal_pos
dx = np.array(animal_pos.iloc[:,0][1:])-np.array(animal_pos.iloc[:,0][:-1]); dx=np.concatenate(([0],dx))
dy = np.array(animal_pos.iloc[:,1][1:])-np.array(animal_pos.iloc[:,1][:-1]); dy=np.concatenate(([0],dx))
#dy = array(data['Y'][1:])-array(data['Y'][:-1]); dy=np.concatenate(([0],dy))


hist_counts,hist_bins=np.histogram(abs(dx))
counts=0
for i in range(len(hist_counts)):
    if counts < len(dx)*0.98:
        counts+=hist_counts[i]
        idx=i
threshold=hist_bins[idx+1]

for i in range(len(dx)):
    if abs(dx[i])>threshold: #Tracking noise threshold
        #data['X'][i]=data['X'][i-1]# NaN#data['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
        data.loc[i]= np.NaN #data['Y'][i-1]


hist_countsy,hist_binsy=np.histogram(abs(dy))
countsy=0
for i in range(len(hist_counts)):
    if counts < len(dy)*0.98:
        countsy+=hist_countsy[i]
        idxy=i
thresholdy=hist_binsy[idx+1]
        
for i in range(len(dy)):
    if abs(dy[i])>threshold: #Tracking noise threshold
        #data['X'][i]=data['X'][i-1]# NaN#data['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
        data.loc[i]= np.NaN #data['Y'][i-1]
       


os.exit()


'''
abs_dx=abs(dx)
        
plt.hist(abs_dx,color='r')
plt.hist(abs_dx[abs_dx < threshold])
'''



#Calculate distance between consecutive x,y points
dx = np.array(animal_pos.iloc[:,0][1:])-np.array(animal_pos.iloc[:,0][:-1]); dx=np.concatenate(([0],dx))
dy = np.array(animal_pos.iloc[:,1][1:])-np.array(animal_pos.iloc[:,1][:-1]); dy=np.concatenate(([0],dx))
#dy = array(data['Y'][1:])-array(data['Y'][:-1]); dy=np.concatenate(([0],dy))


for i,x in enumerate(dx):
    if abs(dx[i])>30: #Tracking noise threshold
        #data['X'][i]=data['X'][i-1]# NaN#data['X'][i-1]  #Assign NaN to position x if computed distance exceeds threshold
        data['Y'][i]= NaN #data['Y'][i-1]
        
fig,az=plt.subplots()
plt.plot(data['X'],data['Y'])
xmin=min(data['X']); xmax=max(data['X'])
ymin=min(data['Y']); ymax=max(data['Y'])
az.invert_yaxis()
xaxis=plot([(xmin+xmax)/2,(xmin+xmax)/2],[ymin,ymax],color='k')
yaxis=plot([xmin,xmax],[(ymax+ymin)/2,(ymax+ymin)/2],color='k')
plt.title('rd1M2-m') 

#Cleans data by removing all NaNs
data=data.dropna()
dx=pd.DataFrame(dx).dropna()
dy=pd.DataFrame(dy).dropna()





#cal

pos_x=np.array(data['X'])
pos_y=np.array(data['Y'])



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


quad_coverage=pd.DataFrame(index=['up_left','up_right','buttom_left','buttom_right'], \
             columns=['tot_dist','time_spent','velocity(unit)'])


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
    tot_time=len(all_dist)/fr 
    vel=tot_dist/tot_time   
    
    
    quad_coverage.iloc[i,0]=tot_dist #Extracting distance covered in quadrant
    quad_coverage.iloc[i,1]=tot_time #Extracting distance covered in quadrant
    quad_coverage.iloc[i,2]=vel #Extracting distance covered in quadrant

        
    
    
    up_left_totDis=sum(up_left_allDis) #distance covered
    up_left_totTime=len(up_left_allDis)/fr  #total time in seconds
    up_left_vel=up_left_totDis/up_left_totTime  #velocity in quadrant
        
