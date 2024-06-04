import numpy as np
import os
import cv2
import glob
import shutil
import argparse

def del_files0(dir_path):
    shutil.rmtree(dir_path)




parser = argparse.ArgumentParser()
parser.add_argument('--start',default=1,type=int)
parser.add_argument('--end',default=2,type=int) 
parser.add_argument('--base_dir',default='./VDW_Demo_Dataset/processed_dataset/', type=str)
parser.add_argument('--deletetxt',default='./check/bad_demo.txt', type=str)
args = parser.parse_args()

    
base_dir = args.base_dir

count_bad = 0
cv = 0
ch = 0
cc = 0
t_hori = 15
for i in range(args.start-1,args.end):

    print(i+1)
    video_dir = base_dir + '%06d'%(i+1)
    if not os.path.exists(video_dir):
        continue
    
    # ratio of vertical disparsity > 2  10%
    ver_ratio_lr = []
    gjr = []
    with open(video_dir + '/ver_ratio.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip('\n')
            gjr.append(float(line))
    ver_ratio_lr.append(gjr[-2])
    ver_ratio_lr.append(gjr[-1])
        
        

    # horizontal disparsity range < 10      
    gjr = []  
    range_lr = []
    with open(video_dir + '/range_avg.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip('\n')
            gjr.append(float(line))
    range_lr.append(gjr[-2])
    range_lr.append(gjr[-1])
    
    # consistency mask ratio  
    con_dir = video_dir + '/flow'
    frames = os.listdir(con_dir)
    frames.sort(key= lambda x:int(x[6:12]))
    if frames[0][-1] == 'g':
 
        img = cv2.imread(os.path.join(con_dir, frames[0])) 
        img_size = (img.shape[1], img.shape[0])
    elif frames[0][-1] == 't':
        img = cv2.imread(os.path.join(con_dir, frames[1]))
        img_size = (img.shape[1], img.shape[0])
    
    all_right = []
    for frame in frames: 
        if frame[-1] == 'g' and frame[-9] == 'c':#frame[-5] == 'c':
            f_path = os.path.join(con_dir, frame)
            #print(f_path)
            image = cv2.imread(f_path,-1)
            ratio = len(image[image==0])/(image.shape[0]*image.shape[1])
            all_right.append(ratio)
        
        
    all_left = []
    for frame in frames: 
        if frame[-1] == 'g' and frame[-5] == 'c':
            f_path = os.path.join(con_dir, frame)
            #print(f_path)
            image = cv2.imread(f_path,-1)
            ratio = len(image[image==0])/(image.shape[0]*image.shape[1])
            all_left.append(ratio)
    
    print(np.mean(all_left))
    
    if ver_ratio_lr[0] > 0.1 or ver_ratio_lr[1] > 0.1 or range_lr[0] < t_hori or range_lr[1] < t_hori or np.mean(all_left) < 0.7 or np.mean(all_right) < 0.7:
        count_bad = count_bad + 1
        with open(args.deletetxt,'a') as f:
            f.write(video_dir + '\n')
        #print('bad-',count_bad,':',video_dir)
    
    if ver_ratio_lr[0] > 0.1 or ver_ratio_lr[1] > 0.1:
        cv = cv + 1
    if range_lr[0] < t_hori or range_lr[1] < t_hori : 
        ch = ch + 1
    
    if np.mean(all_left) < 0.7 or np.mean(all_right) < 0.7:
        cc = cc + 1
 
if count_bad ==0:
    with open(args.deletetxt,'a') as f:
        f.write('All GOOD!' + '\n')

print('vertical_bad:',cv,'horizontal_bad:',ch,'consistency_mask_bad:',cc,'all_bad:',count_bad)
        