# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:48:57 2020

@author: labuser
"""

import pandas as pd
import numpy as np


df = pd.read_csv(r'F:\training_videos\122519\mouseM1-cDLC_resnet50_250framesDec13shuffle1_1030000.csv')
df.head()
 
df.drop(['scorer', 'bodyparts', 'coords', 'objectA', 'likelihood'], axis=1, inplace=True) 


df.drop(to_drop, inplace=True, axis=1)