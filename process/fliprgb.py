import cv2
import os
import glob
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir 
fromdir = base_dir + '/left/'
todir = base_dir + '/left_flip/'
all = os.listdir(fromdir)
for i in range(len(all)):
    img = cv2.imread(fromdir+all[i])
    flip = cv2.flip(img, 1)
    cv2.imwrite(todir+all[i],flip)

fromdir = base_dir + '/right/'
todir = base_dir + '/right_flip/' 
all = os.listdir(fromdir)
for i in range(len(all)):
    img = cv2.imread(fromdir+all[i])
    flip = cv2.flip(img, 1)
    cv2.imwrite(todir+all[i],flip)