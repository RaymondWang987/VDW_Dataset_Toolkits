import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as io
import os
import argparse
import glob
import numpy as np

base_dir = '/data3/wangyiran/dataset/'

for i in range(605,3121):


    video_dir = base_dir + '%06d'%(i+1) + '/'
    
    print(i+1,'/3117:',video_dir)
    
    lgt_dir = video_dir + '/left_gt/'
    rgt_dir = video_dir + '/right_gt/'
    lrgb_dir = video_dir + '/left/'
    rrgb_dir = video_dir + '/right/'
    lm_dir = video_dir + '/left_mask/'
    rm_dir = video_dir + '/right_mask/'
    
    tovideo_dir = '/data1/wangyiran/work2/checkgtvideos/' + '%06d'%(i+1) + '.avi'
    
    fps = 24
    # get frames list
    lgt_frames = os.listdir(lgt_dir)
    lgt_frames.sort(key= lambda x:int(x[-10:-4]))
    rgt_frames = os.listdir(rgt_dir)
    rgt_frames.sort(key= lambda x:int(x[-10:-4]))
    lrgb_frames = os.listdir(lrgb_dir)
    lrgb_frames.sort(key= lambda x:int(x[-10:-4]))
    rrgb_frames = os.listdir(rrgb_dir)
    rrgb_frames.sort(key= lambda x:int(x[-10:-4]))
    lm_frames = os.listdir(lm_dir)
    lm_frames.sort(key= lambda x:int(x[-10:-4]))
    rm_frames = os.listdir(rm_dir)
    rm_frames.sort(key= lambda x:int(x[-10:-4]))
    
    
    # shape
    img = cv2.imread(os.path.join(lgt_dir, lgt_frames[0]))
    img_size = (img.shape[1]*2, img.shape[0]*4)


    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # if want to write .mp4 file, use 'MP4V'
    videowriter = cv2.VideoWriter(tovideo_dir, fourcc, fps, img_size)

    for j in range(len(lgt_frames)):
        
        print('frame:',j+1,'/',len(lrgb_frames))
        
        lrgb_path = os.path.join(lrgb_dir, lrgb_frames[j])
        lrgb = cv2.imread(lrgb_path)
        rrgb_path = os.path.join(rrgb_dir, rrgb_frames[j])
        rrgb = cv2.imread(rrgb_path)
        rgb = np.hstack((lrgb, rrgb))
    
        lgt_path = os.path.join(lgt_dir, lgt_frames[j])
        lgt = cv2.imread(lgt_path)
        rgt_path = os.path.join(rgt_dir, rgt_frames[j])
        rgt = cv2.imread(rgt_path)
        gt = np.hstack((lgt, rgt))
        
        lm_path = os.path.join(lm_dir, lm_frames[j])
        lm = cv2.imread(lm_path)
        rm_path = os.path.join(rm_dir, rm_frames[j])
        rm = cv2.imread(rm_path)
        mask = np.hstack((lm, rm))
        
        rgb_gt = np.vstack((rgb, gt))
        rgb_gt_mask = np.vstack((rgb_gt, mask))
        
        lgt_mask = lgt
        rgt_mask = rgt
        lgt_mask[lm==255] = 0
        rgt_mask[rm==255] = 0
        gt_mask = np.hstack((lgt_mask, rgt_mask))
        rgb_gt_mask_gtmask = np.vstack((rgb_gt_mask,gt_mask))
        
        
        videowriter.write(rgb_gt_mask_gtmask)



    videowriter.release()