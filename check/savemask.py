import os
import cv2
import numpy as np
import shutil
import argparse
def del_files0(dir_path):
    shutil.rmtree(dir_path)
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--start',default=1,type=int)
parser.add_argument('--end',default=2,type=int) 
parser.add_argument('--base_dir',default='./VDW_Demo_Dataset/processed_dataset/', type=str)
args = parser.parse_args()

base_dir = args.base_dir
for i in range(args.start-1,args.end):

    print(i+1)
    video_dir = base_dir + '%06d'%(i+1) + '/'
    
    if not os.path.exists(video_dir[0:-1]):
        continue
    
    lmask_dir = video_dir + 'left_mask/'
    rmask_dir = video_dir + 'right_mask/'
    
    os.makedirs(lmask_dir, exist_ok=True)
    os.makedirs(rmask_dir, exist_ok=True)
    
    flow_dir = video_dir + 'flow/'
    
    frame_num = len(os.listdir(flow_dir))//4
    
    for j in range(frame_num):
        mask_con_l = flow_dir + 'frame_' + '%06d'%(2*(j))+'_occ.png'
        shutil.copy(mask_con_l ,lmask_dir + 'frame_%06d.png'%j) 
        
        mask_con_r = flow_dir + 'frame_' + '%06d'%(2*(j))+'_occ_bwd.png'
        shutil.copy(mask_con_r ,rmask_dir + 'frame_%06d.png'%j) 
    
    #print(flow_dir[0:-1])
    del_files0(flow_dir[0:-1])