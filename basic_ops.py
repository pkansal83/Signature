# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:15:35 2019

@author: pkansal
"""

import os
import cv2
import more_itertools
import pandas as pd
import numpy as np
from math import sqrt
from Preprocessing import preprocessing
from features import all_features

class Similarity(): 

    def euclidean_distance(self,x,y):
        """ return euclidean distance between two lists """
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
    def manhattan_distance(self,x,y):
        """ return manhattan distance between two lists """
        return sum(abs(a-b) for a,b in zip(x,y))


def splitter(img,n_row,n_col):
    img_height, img_width= img.shape
    row_splits = [list(x) for x in more_itertools.divide(n_row, range(img_height))]
    col_splits = [list(x) for x in more_itertools.divide(n_col, range(img_width))]
    img_parts = []    
    for r in range(n_row):
        row_start = row_splits[r][0]
        row_end = row_splits[r][-1]
        for c in range(n_col):
            col_start = col_splits[c][0]
            col_end = col_splits[c][-1]
            img_parts.append(img[row_start:row_end,col_start:col_end])    
    return(img_parts)   

def diff_matrix(data,method):
    m = Similarity()
    met = {'Eucledian' : m.euclidean_distance,
           'Manhattan' : m.manhattan_distance}
    out = pd.DataFrame(index = data.index,columns = data.index)
    for i in data.index:
        for j in data.index:
            out.loc[i,j] = met[method](data.loc[i,:],data.loc[j,:])
    return(out)

def feature(file,split=None):
    image = cv2.imread(file)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p_image = preprocessing(grey)
    if split != None:
        instance = splitter(p_image,split[0],split[1])
        features = pd.DataFrame([all_features(np.array(inst),np.array(inst)) for inst in instance])
        features.index = [os.path.basename(file).replace('.png','')+'_'+str(k) for k in range(1,split[0]*split[1]+1)]
    else:
        features = pd.DataFrame([all_features(np.array(p_image),np.array(p_image))])
        features.index = [os.path.basename(file).replace('.png','')]
    return(features)

