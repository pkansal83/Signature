# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:09:06 2019

@author: pkansal
"""

import os
import shutil
import cv2
from itertools import combinations

import sys
sys.path.append(os.getcwd())
from Preprocessing import preprocessing
from basic_ops import feature, diff_matrix
import numpy as np
import pandas as pd
from statistics import median
from scipy.stats import iqr

def mad(data):
    return np.mean(np.absolute(data - np.mean(data)))

class train_data_prep:
    # declaring the tile_combinations, distance_metric, number_of_signatures and number_of_copies_per_sign
    def __init__(self,files,tile_comb,dist_met):
        self.files = files        
        self.tile_comb = tile_comb
        self.dist_met = dist_met
        self.num_sign = len(files)       # number of original variants received for a sign
                
    # splitting the received variants into train and test on the basis of similarity
    def train_test_split(self,dif_mat):
        comb = list(combinations(dif_mat.columns,self.num_sign-1))
        v = np.Inf
        for i in comb:
            sub_data = np.array(dif_mat.loc[i,i])
            v1 = np.var(sub_data[np.triu_indices(self.num_sign-1,k=1)])
            if v1 < v:
                v = v1
                self.train_set = ''.join([str(int(j)) for j in i])
                self.test_img = str(int(dif_mat.columns.difference(i).values[0]))
        return((self.train_set,self.test_img))
    
    def testing(self,comb):
        train_s = list(self.train_set)
        num_tiles = comb[0] * comb[1]
        Data = pd.DataFrame()
        for file in self.files:
            if str(int(os.path.basename(file).rstrip('.png'))) in train_s:
                Data = pd.concat([Data,feature(file,comb)],axis=0)
            else:
                test_feat = feature(file,comb)
                
        rang = pd.DataFrame(self.range_create(comb)['range_mat']).T
        rang.columns = ['Median','IQR']
        flag = []
        red_flag = []
        for i in range(1,num_tiles+1):
            test_ind = [x for x in test_feat.index if x.endswith('_'+str(i))]
            train_ind = [x for x in Data.index if x.endswith('_'+str(i))]
            rang_ind = [x for x in rang.index if x.endswith('_'+str(i))]
            val = [diff_matrix(Data.loc[[j],:].append(test_feat.loc[test_ind,:]).reset_index(drop=True),self.dist_met).iloc[0,1] for j in train_ind]
            scr = sum([1 if x <= (rang.loc[rang_ind[0],'Median']+0.5*rang.loc[rang_ind[0],'IQR']) else 0 for x in val])
            flag.append(scr)
            red_flag.append(0 if scr > 2 else 1)
        score = 100*(sum([x*(1-y) for x,y in zip(flag,red_flag)])/(Data.shape[0]) + (num_tiles - sum(red_flag))/num_tiles)/2
        return({'flag':flag, 'red_flag':red_flag, 'score':score})

    def best_comb(self):
        scr = {i:self.testing(i)['score'] for i in self.tile_comb}
        opt_comb = max(scr, key=lambda key: scr[key])
        return(opt_comb)
        
    def range_create(self,comb):
        train_s = list(self.train_set)
        num_tiles = comb[0] * comb[1]
        Data = pd.DataFrame()
        for file in self.files:
            if str(int(os.path.basename(file).rstrip('.png'))) in train_s:
                Data = pd.concat([Data,feature(file,comb)],axis=0)
        out = {}
        for tile in range(1,num_tiles+1):
            col = [x for x in Data.index if x.endswith('_'+str(tile))]
            subdata = Data.loc[col,:]
            dif_mat = np.array(diff_matrix(subdata,self.dist_met))
            v1 = dif_mat[np.triu_indices(self.num_sign-1,k=1)]
            Median = median(v1)
            IQR = iqr(v1)
            out['Tile'+'_'+str(tile)] = [Median,IQR]
        return({'feat_mat':Data, 'range_mat':out})

    
def train_files(src_path,cust_id):
    
    tile_comb = ((2,2),(2,3),(3,2),(2,4),(3,3),(3,4))
    dist_met = "Manhattan"      # any one out of ['Manhattan','Eucledian','Cosine','Chebyshev','Percent_error']
                    
    cust_i = train_data_prep([os.path.join(src_path,cust_id,"original",i) for i in os.listdir(os.path.join(src_path,cust_id,"original"))],tile_comb,dist_met)

    feat_mat = pd.DataFrame()
    for file in cust_i.files:
        feat_mat = pd.concat([feat_mat,feature(os.path.join(src_path,cust_id,"original",file))],axis=0)
    feat_mat.T.to_csv(os.path.join(src_path,cust_id,'features_matrix.csv'))

    dif_mat = diff_matrix(feat_mat,dist_met)
    dif_mat.to_csv(os.path.join(src_path,cust_id,'similarity.csv'))

    train, test = cust_i.train_test_split(dif_mat)
    best_comb = cust_i.best_comb()
    
    best_comb_feat = cust_i.range_create(best_comb)['feat_mat']
    best_comb_feat.to_csv(os.path.join(src_path,cust_id,'best_comb_features_matrix.csv'))

    range_df = pd.DataFrame(cust_i.range_create(best_comb)['range_mat']).T
    range_df.columns = ['Median','IQR']
    range_df.to_csv(os.path.join(src_path,cust_id,'range.csv'))
    
    with open(os.path.join(src_path,cust_id,'best_comb.txt'), "w") as text_file:
        text_file.write(str(best_comb))


def single_user_add(img_paths, out_path, cust_id=""):
    
    if not (isinstance(img_paths,list) or isinstance(img_paths,tuple)):
        return("Error")
        
    if len(img_paths) < 4:
        return("Select atleast 4 Signatures")
        
    if cust_id == "":
        cust_id = "new_user"
        i = 1
        while cust_id in os.listdir(out_path):
            cust_id = "new_user_" + str(i).zfill(2)
            i = i+1
        del i
            
    if os.path.isdir(os.path.join(out_path,cust_id)):
        shutil.rmtree(os.path.join(out_path,cust_id))
        
    os.mkdir(os.path.join(out_path,cust_id))
    os.mkdir(os.path.join(out_path,cust_id,"original"))
    os.mkdir(os.path.join(out_path,cust_id,"processed"))
    
    for i in range(len(img_paths)):
        shutil.copy(img_paths[i],os.path.join(out_path,cust_id,"original",str(i+1).zfill(2)+".png"))
        image = cv2.imread(img_paths[i])
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p_image = preprocessing(grey)
        cv2.imwrite(os.path.join(out_path,cust_id,"processed",str(i+1).zfill(2)+".png"),p_image)
    
    train_files(out_path,cust_id)
    return(cust_id)


def multi_user_add(src_path, out_path):
    #Reading all images from source Path
    cust = {}
    for i in src_path:
        j = os.path.basename(i).rsplit('_',1)[0]
        if j not in cust:
            cust[j] = []
        cust[j].append(i)

    cnt = 0
    replaced = 0
    drop = 0
    for cust_id in cust:
        if len(cust[cust_id]) < 4:
            drop = drop+1
            continue
                            
        if os.path.isdir(os.path.join(out_path,cust_id)):
            replaced = replaced+1
            shutil.rmtree(os.path.join(out_path,cust_id))
            
        os.mkdir(os.path.join(out_path,cust_id))
        os.mkdir(os.path.join(out_path,cust_id,"original"))
        os.mkdir(os.path.join(out_path,cust_id,"processed"))
        
        for i in range(len(cust[cust_id])):
            shutil.copy(cust[cust_id][i],os.path.join(out_path,cust_id,"original",str(i+1).zfill(2)+".png"))
            image = cv2.imread(cust[cust_id][i])
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            p_image = preprocessing(grey)
            cv2.imwrite(os.path.join(out_path,cust_id,"processed",str(i+1).zfill(2)+".png"),p_image)
        
        train_files(out_path,cust_id)
        cnt = cnt+1
    return([cnt-replaced,replaced,drop])
        
        
def rm_users(data_path,cust_ids):
    if isinstance(cust_ids,str):
        cust_ids = [cust_ids]
    for i in cust_ids:
        if os.path.isdir(os.path.join(data_path,i)):
            shutil.rmtree(os.path.join(data_path,i))
