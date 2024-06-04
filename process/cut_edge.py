import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import argparse


def remove_the_blackborder(image):

    image = cv2.imread(image)      
    #print(image.shape)
    img = cv2.medianBlur(image, 5) 
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY) 
    binary_image = b[1]            
    binary_image = cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
 
    edges_y, edges_x = np.where(binary_image==255) 
    bottom = min(edges_y)+5             
    top = max(edges_y)-5 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    #res_image = image[bottom+5:bottom+height-5, left+5:left+width-5]


    return bottom,top,left,right                                          


def cut(images,bottom,top,left,right):
    for i in range(len(images)):
        print('cutting:',i)
        image = cv2.imread(images[i])       
        image = image[bottom:top,left:right,:]                       
        cv2.imwrite(images[i], image)
        
        
        
parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir 
'''                          
save_path = "./"                                   
if not os.path.exists(save_path):
    os.mkdir(save_path)
'''
# bottom,top,left,right = 140,940,20,1900 
bottom,top,left,right = 140,1080-140-80,20,1900

cut(glob.glob(base_dir + '/left/*.png'),bottom,top,left,right)
cut(glob.glob(base_dir + '/right/*.png'),bottom,top,left,right)

