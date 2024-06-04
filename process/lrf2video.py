import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as io
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir 

im_dir = base_dir + '/left/'#'/data1/wangyiran/mytrans/plt/'#'/data1/wangyiran/mytrans/firstmae/asee'
video_dir = base_dir + '/leftrgb.avi'#'/data1/wangyiran/mytrans/plt/2048.avi'#'/data1/wangyiran/mytrans/firstmae/asee/auau.avi'
# set saved fps
fps = 24
# get frames list
frames = os.listdir(im_dir)
frames.sort(key= lambda x:int(x[-10:-4]))


# w,h of image
if frames[0][-1] == 'g':
    img = cv2.imread(os.path.join(im_dir, frames[0]))
    img_size = (img.shape[1], img.shape[0])
elif frames[0][-1] == 't':
    img = cv2.imread(os.path.join(im_dir, frames[1]))
    img_size = (img.shape[1], img.shape[0])


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames: 
        if frame[-1] == 'g':
            f_path = os.path.join(im_dir, frame)
            image = cv2.imread(f_path)
            #image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            videowriter.write(image)
            print(frame + " has been written!")



videowriter.release()


im_dir = base_dir + '/left_flip/'#'/data1/wangyiran/mytrans/plt/'#'/data1/wangyiran/mytrans/firstmae/asee'
video_dir = base_dir + '/leftrgb_flip.avi'#'/data1/wangyiran/mytrans/plt/2048.avi'#'/data1/wangyiran/mytrans/firstmae/asee/auau.avi'
# set saved fps
fps = 24
# get frames list
frames = os.listdir(im_dir)
frames.sort(key= lambda x:int(x[-10:-4]))


# w,h of image
if frames[0][-1] == 'g':
    img = cv2.imread(os.path.join(im_dir, frames[0]))
    img_size = (img.shape[1], img.shape[0])
elif frames[0][-1] == 't':
    img = cv2.imread(os.path.join(im_dir, frames[1]))
    img_size = (img.shape[1], img.shape[0])


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames: 
        if frame[-1] == 'g':
            f_path = os.path.join(im_dir, frame)
            image = cv2.imread(f_path)
            #image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            videowriter.write(image)
            print(frame + " has been written!")



videowriter.release()

im_dir = base_dir + '/right/'#'/data1/wangyiran/mytrans/plt/'#'/data1/wangyiran/mytrans/firstmae/asee'
video_dir = base_dir + '/rightrgb.avi'#'/data1/wangyiran/mytrans/plt/2048.avi'#'/data1/wangyiran/mytrans/firstmae/asee/auau.avi'
# set saved fps
fps = 24
# get frames list
frames = os.listdir(im_dir)
frames.sort(key= lambda x:int(x[-10:-4]))


# w,h of image
if frames[0][-1] == 'g':
    img = cv2.imread(os.path.join(im_dir, frames[0]))
    img_size = (img.shape[1], img.shape[0])
elif frames[0][-1] == 't':
    img = cv2.imread(os.path.join(im_dir, frames[1]))
    img_size = (img.shape[1], img.shape[0])


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames: 
        if frame[-1] == 'g':
            f_path = os.path.join(im_dir, frame)
            image = cv2.imread(f_path)
            #image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            videowriter.write(image)
            print(frame + " has been written!")



videowriter.release()

im_dir = base_dir + '/right_flip/'#'/data1/wangyiran/mytrans/plt/'#'/data1/wangyiran/mytrans/firstmae/asee'
video_dir = base_dir + '/rightrgb_flip.avi'#'/data1/wangyiran/mytrans/plt/2048.avi'#'/data1/wangyiran/mytrans/firstmae/asee/auau.avi'
# set saved fps
fps = 24
# get frames list
frames = os.listdir(im_dir)
frames.sort(key= lambda x:int(x[-10:-4]))


# w,h of image
if frames[0][-1] == 'g':
    img = cv2.imread(os.path.join(im_dir, frames[0]))
    img_size = (img.shape[1], img.shape[0])
elif frames[0][-1] == 't':
    img = cv2.imread(os.path.join(im_dir, frames[1]))
    img_size = (img.shape[1], img.shape[0])


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames: 
        if frame[-1] == 'g':
            f_path = os.path.join(im_dir, frame)
            image = cv2.imread(f_path)
            #image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            videowriter.write(image)
            print(frame + " has been written!")



videowriter.release()

