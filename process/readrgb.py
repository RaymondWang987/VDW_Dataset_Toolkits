import cv2
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir 
root_left_dir = base_dir + 'left/'
root_right_dir = base_dir + 'right/'
root_save_dir = base_dir + 'rgblr/'
left = sorted(os.listdir(root_left_dir))
right = sorted(os.listdir(root_right_dir))

for i in range(len(left)):
    left_img = cv2.imread(root_left_dir+left[i])
    right_img = cv2.imread(root_right_dir+right[i])
    print(left[i],right[i])
    cv2.imwrite(root_save_dir+'frame_'+str(2*i).zfill(6)+'.png',left_img)
    cv2.imwrite(root_save_dir+'frame_'+str(2*i+1).zfill(6)+'.png',right_img) 

