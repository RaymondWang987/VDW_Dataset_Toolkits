import os
import glob
import torch
import cv2
import argparse
import time

ffmpeg = "/home/wangyiran/anaconda3/envs/rcvd/bin/ffmpeg"
ffprobe = "/home/wangyiran/anaconda3/envs/rcvd/bin/ffprobe"


def mkdir_ifnotexists(dir):
    if os.path.exists(dir):
        return
def extract_frames(video_file,path):
        frame_dir = path #+"/input_png" 
        mkdir_ifnotexists(frame_dir)

        if not os.path.exists(video_file):
            sys.exit("ERROR: input video file '%s' not found.", video_file)

        cmd = "%s -i %s -start_number 0 -vsync 0 %s/frame_%%06d.png" % (
            ffmpeg,
            video_file,
            frame_dir,
        )
        print(cmd)
        os.popen(cmd).read()

parser = argparse.ArgumentParser()

parser.add_argument('-bd', '--base_dir')
args = parser.parse_args()
base_dir = args.base_dir 
extract_frames(base_dir + 'rgbl.mp4',base_dir + '/left/') 
 
extract_frames(base_dir +'rgbr.mp4',base_dir + '/right/')  