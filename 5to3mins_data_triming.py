# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:57:18 2021

@author: labuser
"""



import os
import glob
import pandas as pd

os.chdir(r'E:\Edith\Behavior\Data\sighted_size\Sighted-c-excel')
allFiles = glob.glob("*.xlsx") # match your csvs
for file in allFiles:
   df = pd.read_excel(file, encoding='latin-1')
   df = df.iloc[:3300,] # read from row 34 onwards.
   df.to_excel(file)
#   print(f"{file} has removed rows 0-33")



