# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:43:12 2019

@author: pkansal
"""

import cv2
import numpy as np
from scipy import signal

def resizing(image):
    image = cv2.resize(image, (257,109)) 
    return(image)

def crop(img):
    rect = cv2.minAreaRect(cv2.findNonZero(img))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, z, w = min(box.T[0]), min(box.T[1]), max(box.T[0]), max(box.T[1])
    return img[y:w, x:z]

def deskew(gray):
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
    	angle = -(90 + angle)
    else:
    	angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return(rotated)

def binary(image):
    ret,thresh1 = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    return(thresh1)

def thinning(image):
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(image,kernel,iterations = 1)
    return(erosion)

def denoising(image,filt):
    if filt=="median":
        bg = signal.medfilt2d(image, 11)
        mask = image < bg - 0.1
        out=np.where(mask,image,1.0)
    else:
        out=cv2.GaussianBlur(image,(3,3),0)        
    return(out)

def preprocessing(image):

    #Binary
    binary_image=binary(image)

    #Angle correction
    binary_image=cv2.bitwise_not(binary_image)
    deskew_image=deskew(binary_image)  

    #Image Cropping
    croped_image=crop(deskew_image)    
    croped_image=cv2.bitwise_not(croped_image)
    croped_image=binary(croped_image)

    #Thinning
    #thinning_image=thinning(croped_image)

    #Denosing
    #denoised_image=denoising(thinning_image,"gaussian")
    #denoised_image=binary(denoised_image)
    
            
    #Resizing(if required)
    #resized=resizing(thinning_image)
    
    return(croped_image)
