# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:43:38 2021

@author: labuser
"""

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import skvideo.io
import skimage.transform as transform
import argparse
from scipy.io import loadmat
import cv2
from tifffile import imsave

fn1 = r"F:\calcium_imaging\Edith\widefield_over_skull\retinotopic_mapping\EC018\2021-03-03\wdf\wdf\wdf_000_a11.mj2"

if __name__ == '__main__':
    ### Parser ###
    parser = argparse.ArgumentParser()
    parser.add_argument("fn", help="File path to video file")
    args = parser.parse_args()
    fn1 = args.fn1
    ### Parser ###

    dsize = 1
    smax = 50
    s = 10
    outputparameters = {'-pix_fmt' : 'gray16be'}     # specify to import as uint16, otherwise it's uint8

    # Import video
    vid = np.squeeze(skvideo.io.vread(fn1, outputdict=outputparameters))
    base = np.nanmean(vid[0:10], axis=0)
   
    imsave('EC018.tif', vid)
   
    
fn = r"F:\calcium_imaging\Edith\widefield_over_skull\retinotopic_mapping\EC018\2021-03-03\wdf\wdf\wdf_000_a11.mj2_events"
def sbx_get_ttlevents(fn):
    '''Load TTL events from scanbox events file.
       Based on sbx_get_ttlevents.m script.

       INPUT
       fn   : Filepath to events file

       OUTPUT
       evt  : List of TTL trigger time in units of frames (numpy.array)
    '''
    data = loadmat(fn)['ttl_events']
    if not(data.size == 0):
        evt = data[:,2]*256 + data[:,1]     # Not sure what this does
    else:
        evt = np.array([])

    return evt

  
  
  start = 0
  ttl = 177
  trial =[]
  for i in range(10):
    trial.append(ttl*i+start)
  print(trial)
  matrix = sum(trial)/10
  print(matrix)







