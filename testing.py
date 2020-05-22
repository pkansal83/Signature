# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 01:23:08 2019

@author: pkansal
"""

import cv2
import numpy as np
import pandas as pd
import os
from Preprocessing import preprocessing
from basic_ops import splitter, feature, diff_matrix

def test_img(img,split_comb,flag):
    alpha = 0.5
    red = (0,0,255)
    green = (0,128,0)
    yellow = (0,255,255)
    orange = (0,255,165)
    image = cv2.imread(img)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p_image = preprocessing(grey)
    n_row,n_col = split_comb
    instance = splitter(p_image,n_row,n_col)
    new_im = []
    for i in range(len(flag)):
        backtorgb = cv2.cvtColor(instance[i],cv2.COLOR_GRAY2RGB)
        overlay = backtorgb.copy()
        output = backtorgb.copy()
        if flag[i] <= 1:
            cv2.rectangle(overlay, (0, 0), (174, 64), red, -1)
        elif flag[i] == 2:
            cv2.rectangle(overlay, (0, 0), (174, 64), orange, -1)
        elif flag[i] == 3:
            cv2.rectangle(overlay, (0, 0), (174, 64), yellow, -1)
        else:
            cv2.rectangle(overlay, (0, 0), (174, 64), green, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        outputImage = cv2.copyMakeBorder(output,1,1,1,1,cv2.BORDER_CONSTANT,value=(0,0,0))
        new_im.append(outputImage)
        
    out_im = np.concatenate([np.concatenate(new_im[(i*n_col):((i+1)*n_col)],axis=1) for i in range(n_row)],axis=0)
    return(out_im)

def testing(cust,train_path,img_path,method,p=0.5):
    best_comb = open(os.path.join(train_path,cust,'best_comb.txt')).read()
    comb = tuple(map(int,best_comb.strip('(|)').split(",")))
    new_im = feature(img_path,split=comb)
    train_feat = pd.read_csv(os.path.join(train_path,cust,'best_comb_features_matrix.csv'),index_col=0)
    rang = pd.read_csv(os.path.join(train_path,cust,'range.csv'),index_col=0)
    num_tiles = comb[0]*comb[1]
    flag = []
    red_flag = []
    for i in range(1,num_tiles+1):
        test_ind = [x for x in new_im.index if x.endswith('_'+str(i))]
        train_ind = [x for x in train_feat.index if x.endswith('_'+str(i))]
        rang_ind = [x for x in rang.index if x.endswith('_'+str(i))]
        val = [diff_matrix(train_feat.loc[[j],:].append(new_im.loc[test_ind,:]).reset_index(drop=True),method).iloc[0,1] for j in train_ind]
        scr = sum([1 if x <= (rang.loc[rang_ind[0],'Median']+p*rang.loc[rang_ind[0],'IQR']) else 0 for x in val])
        flag.append(scr)
        red_flag.append(0 if scr > 2 else 1)
    score = 100*(sum([x*(1-y) for x,y in zip(flag,red_flag)])/(train_feat.shape[0]) + (num_tiles - sum(red_flag))/num_tiles)/2
    cv2.imwrite(os.path.join(train_path,cust,'test.png'),test_img(img_path,comb,flag))
    return(score)
