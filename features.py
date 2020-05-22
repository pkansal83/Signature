# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:43:12 2019

@author: pkansal
"""

import cv2
import numpy as np
import pandas as pd

def horizontal_splitter(gray):
    height, width = gray.shape[:2]
    split_point = int(height/2)
    upper_gray = gray[:split_point,:]
    lower_gray = gray[split_point:,:]
    return(upper_gray,lower_gray)
    
def diagonal_splitter_upper1(p):
    height, width = p.shape[:2]
    pt1 = (width,0)
    pt2 = (width,height)
    pt3 = (0,height)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(p, [triangle_cnt], 0, (255,255,255), -1)
    n_black_pix = np.sum(p == 0)
    return(n_black_pix)

def diagonal_splitter_lower1(p):
    height, width = p.shape[:2]
    pt1 = (width,0)
    pt2 = (0,height)
    pt3 = (0,0)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(p, [triangle_cnt], 0, (255,255,255), -1)
    n_black_pix = np.sum(p == 0)
    return(n_black_pix)   

def diagonal_splitter_upper2(p):
    height, width = p.shape[:2]
    pt1 = (width,height)
    pt2 = (0,height)
    pt3 = (0,0)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(p, [triangle_cnt], 0, (255,255,255), -1)
    n_black_pix = np.sum(p == 0)
    return(n_black_pix)

def diagonal_splitter_lower2(p):
    height, width = p.shape[:2]
    pt1 = (width,height)
    pt2 = (0,0)
    pt3 = (width,0)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(p, [triangle_cnt], 0, (255,255,255), -1)
    n_black_pix = np.sum(p == 0)
    return(n_black_pix)   

def vertical_splitter(gray):
    height, width = gray.shape[:2]
    split_point = int(width/2)
    left_gray = gray[:,:split_point]
    right_gray = gray[:,split_point:width]
    return(left_gray, right_gray)
    
def binary(img):
    ret,thresh1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    return(thresh1)

def cog(gray):
    binary_img = binary(gray)
    m = np.zeros(binary_img.shape)
    m[binary_img == 0] = 1

    x = []
    y = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if (m[i,j] == 1):
                x.append(i)
                y.append(j)

    avg_x = (np.sum(x)/len(x))/m.shape[0]
    avg_y = (np.sum(y)/len(y))/m.shape[1]
    
    if pd.isnull(avg_x):
        avg_x = 0
    if pd.isnull(avg_y):
        avg_y = 0
    
    return (avg_x,avg_y)

def slope(gray):
    left_gray, right_gray = vertical_splitter(gray)
    avg_x_left, avg_y_left = cog(left_gray)
    avg_x_right, avg_y_right = cog(right_gray)
    slope = (avg_y_left - avg_y_right)/(avg_x_left - avg_x_right)
    return(slope)
    
def High_Gray_level(gray_1):
    mark=gray_1.tolist()
    m1=[]
    m2=[]
    for j in range(0,len(mark)):
        k=0
        k2=0
        for i in range(0,len(mark[j])):
            k=k+((i*i)*mark[j][i])
            k2=k2+mark[j][i]
        #print(j)
        m1.append(k)
        m2.append(k2)
                
    feature=np.round(sum(m2)/sum(m1),4)
    return(feature)

def canny_edge_detection(gray):
    #blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(gray, 30, 150)
    return(edged)

def extract_features(gray, image_type, orig,org_image):
        
    ############### Height to Width Ratio (F1)
    height, width = gray.shape[:2]
    h_w_ratio = height/width
    
    ############### Occupancy Ratio (F2)
    if orig == 1:
        n_white_pix = np.sum(gray > 210)
        n_black_pix = np.sum(gray < 45)
        occ_ratio = n_black_pix/(n_white_pix + n_black_pix)
    else:
        n_white_pix = np.sum(gray > 210)
        n_black_pix = np.sum(gray < 45)
        occ_ratio = n_white_pix/(n_white_pix + n_black_pix)
        
    ############### Density Ratio (F3)
    left_gray, right_gray = vertical_splitter(gray)
    n_black_pix_left = np.sum(left_gray < 45)
    n_black_pix_right = np.sum(right_gray < 45) 
    density_ratio1 = n_black_pix_left/(n_black_pix_left + n_black_pix_right)
    if pd.isnull(density_ratio1):
        density_ratio1 = 0

    upper_gray, lower_gray = horizontal_splitter(gray)
    n_black_pix_upper = np.sum(upper_gray < 45)
    n_black_pix_lower = np.sum(lower_gray < 45) 
    density_ratio2 = n_black_pix_upper/(n_black_pix_lower + n_black_pix_upper)
    if pd.isnull(density_ratio2):
        density_ratio2 = 0

    density_ratio3 = diagonal_splitter_upper2(gray.copy())/(diagonal_splitter_lower2(gray.copy()) + diagonal_splitter_upper2(gray.copy()))
    if pd.isnull(density_ratio3):
        density_ratio3 = 0

    density_ratio4 = diagonal_splitter_upper1(gray.copy())/(diagonal_splitter_lower1(gray.copy()) + diagonal_splitter_upper1(gray.copy()))
    if pd.isnull(density_ratio4):
        density_ratio4 = 0
    
    ############### Critical points (F4)
     # find Harris corners
    if orig == 1:
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray_float,np.float32(centroids),(1,1),(-1,-1),criteria)
        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)
        no_of_corners = len(res)/100
         
    else:
        no_of_corners = 0
     
    ############### Center of Gravity (F5)
    if orig == 1:
        avg_x,avg_y = cog(gray)
    else:
        avg_x = 0
        avg_y = 0
            
    ############### Histogram features
    data = gray.ravel()
    mean_hist = data.mean()/255
    var_hist = data.var()/(255*255)
        
    if(orig == 0):
        featureset =  {
                       'h_w_ratio' + image_type      : h_w_ratio,
                       'occ_ratio' + image_type      : occ_ratio,
                       'density_ratio1' + image_type  : density_ratio1,
                       'density_ratio2' + image_type  : density_ratio2,
                       'density_ratio3' + image_type  : density_ratio3,
                       'density_ratio4' + image_type  : density_ratio4,
                       'mean_hist' + image_type      : mean_hist,
                       'var_hist' + image_type       : var_hist}
    else:
        featureset =  {#'h_w_ratio' + image_type      : h_w_ratio,
                       'occ_ratio' + image_type      : occ_ratio,
                       'density_ratio1' + image_type  : density_ratio1,
                       'density_ratio2' + image_type  : density_ratio2,
                       'density_ratio3' + image_type  : density_ratio3,
                       'density_ratio4' + image_type  : density_ratio4,
                       'no_of_corners' + image_type  : no_of_corners,
                       'avg_x' + image_type          : avg_x,
                       'avg_y' + image_type          : avg_y,
                       'mean_hist' + image_type      : mean_hist,
                       'var_hist' + image_type       : var_hist}
    return(featureset)
    
def all_features(gray,orignal_image):
    orig_img_features = extract_features(gray, '_orig', orig = 1,org_image=orignal_image)
    canny = canny_edge_detection(gray)
    canny_features = extract_features(canny, '_canny', orig = 0,org_image=orignal_image)
    final_featureset = orig_img_features.copy()
    final_featureset.update(canny_features)
    return(final_featureset)
