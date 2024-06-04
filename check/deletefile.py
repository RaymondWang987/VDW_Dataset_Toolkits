import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start',default=1,type=int)
parser.add_argument('--end',default=2,type=int) 
parser.add_argument('--base_dir',default='./VDW_Demo_Dataset/processed_dataset/', type=str)
args = parser.parse_args()



def del_files0(dir_path):
    shutil.rmtree(dir_path)
def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)
    print('ok')



base_dir = args.base_dir
for i in range(args.start-1,args.end):
    video_dir = base_dir + '%06d'%(i+1) + '/'
    print(video_dir)
    if not os.path.exists(video_dir[:-1]):
        continue
    if os.path.exists(video_dir+'l1'): 
        del_files0(video_dir + 'l1')
    if os.path.exists(video_dir+'l2'): 
        del_files0(video_dir + 'l2')
    if os.path.exists(video_dir+'l3'): 
        del_files0(video_dir + 'l3')
    if os.path.exists(video_dir+'l4'): 
        del_files0(video_dir + 'l4')
    if os.path.exists(video_dir+'r1'): 
        del_files0(video_dir + 'r1')
    if os.path.exists(video_dir+'r2'): 
        del_files0(video_dir + 'r2')
    if os.path.exists(video_dir+'r3'): 
        del_files0(video_dir + 'r3')
    if os.path.exists(video_dir+'r4'): 
        del_files0(video_dir + 'r4')
    if os.path.exists(video_dir+'left_flip'): 
        del_files0(video_dir + 'left_flip')
    if os.path.exists(video_dir+'left_seg'):
        del_files0(video_dir + 'left_seg')
    if os.path.exists(video_dir+'left_sky'): 
        del_files0(video_dir + 'left_sky')
    if os.path.exists(video_dir+'right_flip'): 
        del_files0(video_dir + 'right_flip')
    if os.path.exists(video_dir+'right_seg'):
        del_files0(video_dir + 'right_seg')
    if os.path.exists(video_dir+'right_sky'): 
        del_files0(video_dir + 'right_sky')
    if os.path.exists(video_dir+'rgblr'): 
        del_files0(video_dir + 'rgblr')
    if os.path.exists(video_dir+'rgbl.mp4'): 
        del_files(video_dir + 'rgbl.mp4')
    if os.path.exists(video_dir+'rgbr.mp4'): 
        del_files(video_dir + 'rgbr.mp4')
    if os.path.exists(video_dir+'leftrgb.avi'): 
        del_files(video_dir + 'leftrgb.avi')
    if os.path.exists(video_dir+'leftrgb.avi'): 
        del_files(video_dir + 'leftrgb.avi')
    if os.path.exists(video_dir+'leftrgb_flip.avi'): 
        del_files(video_dir + 'leftrgb_flip.avi')
    if os.path.exists(video_dir+'rightrgb_flip.avi'): 
        del_files(video_dir + 'rightrgb_flip.avi')
    if os.path.exists(video_dir+'rightrgb.avi'): 
        del_files(video_dir + 'rightrgb.avi')