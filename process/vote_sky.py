import cv2
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir  

v1_dir = base_dir + '/l1/'
v2_dir = base_dir + '/l2/'
v3_dir = base_dir + '/l3/'
v4_dir = base_dir + '/l4/' 

all = os.listdir(v1_dir)

for i in range(len(all)):
    print(v1_dir+all[i])

    v1 = np.array(cv2.imread(v1_dir+all[i],-1),dtype = np.int16)
    v2 = np.array(cv2.imread(v2_dir+all[i],-1),dtype = np.int16)
    v3 = np.array(cv2.imread(v3_dir+all[i],-1),dtype = np.int16)
    v4 = np.array(cv2.imread(v4_dir+all[i],-1),dtype = np.int16)

    vfinal = v1 + v2 + v3 + v4

    vfinal[vfinal<=2*255] = 0
    vfinal[vfinal>=3*255] = 255
    vfinal = vfinal.astype('uint8')

    cv2.imwrite(base_dir+'/left_sky/'+all[i],vfinal) 
    

v1_dir = base_dir + '/r1/'
v2_dir = base_dir + '/r2/'
v3_dir = base_dir + '/r3/'
v4_dir = base_dir + '/r4/' 

all = os.listdir(v1_dir)

for i in range(len(all)):
    print(v1_dir+all[i])

    v1 = np.array(cv2.imread(v1_dir+all[i],-1),dtype = np.int16)
    v2 = np.array(cv2.imread(v2_dir+all[i],-1),dtype = np.int16)
    v3 = np.array(cv2.imread(v3_dir+all[i],-1),dtype = np.int16)
    v4 = np.array(cv2.imread(v4_dir+all[i],-1),dtype = np.int16)

    vfinal = v1 + v2 + v3 + v4

    vfinal[vfinal<=2*255] = 0
    vfinal[vfinal>=3*255] = 255
    vfinal = vfinal.astype('uint8')

    cv2.imwrite(base_dir+'/right_sky/'+all[i],vfinal)    