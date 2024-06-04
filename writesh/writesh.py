import os
import shutil
import argparse


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--shname',type=str)
    parser.add_argument('--cuda',type=str)
    parser.add_argument('--fromdir',type=str)
    parser.add_argument('--todir',type=str)
    parser.add_argument('--cut_black_bar', type=str2bool, default=False)
    args = parser.parse_args()
    fromdir = args.fromdir
    todir = args.todir


    videolist=sorted(os.listdir(fromdir))
    video_intindexlist=[]
    for videoname in videolist:
        video_intindexlist.append(int(videoname.split('.')[0]))

    for i in range(args.start-1,args.end):
        if i+1 not in video_intindexlist:
            continue
        base_dir = todir+'%06d'%(i+1)+'/' 
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(base_dir+'l1', exist_ok=True)
        os.makedirs(base_dir+'l2', exist_ok=True)
        os.makedirs(base_dir+'l3', exist_ok=True)
        os.makedirs(base_dir+'l4', exist_ok=True)
        os.makedirs(base_dir+'r1', exist_ok=True)
        os.makedirs(base_dir+'r2', exist_ok=True)
        os.makedirs(base_dir+'r3', exist_ok=True)
        os.makedirs(base_dir+'r4', exist_ok=True)
        os.makedirs(base_dir+'left', exist_ok=True)
        os.makedirs(base_dir+'left_flip', exist_ok=True)
        os.makedirs(base_dir+'right', exist_ok=True)
        os.makedirs(base_dir+'right_flip', exist_ok=True)
        os.makedirs(base_dir+'rgblr', exist_ok=True)
        os.makedirs(base_dir+'flow', exist_ok=True)
        os.makedirs(base_dir+'left_sky', exist_ok=True)
        os.makedirs(base_dir+'right_sky', exist_ok=True)
        os.makedirs(base_dir+'left_gt', exist_ok=True)
        os.makedirs(base_dir+'right_gt', exist_ok=True)


    # conda init @untested
    os.system("cat template_conda.sh > {}".format(args.shname))

    for i in range(args.start-1,args.end):
        if i+1 not in video_intindexlist:
            continue
        shutil.copy(fromdir+'/'+'%06d'%(i+1)+'.mp4',todir+'/'+'%06d'%(i+1)+'/rgb.mp4') 


    for i in range(args.start-1,args.end):
        if i+1 not in video_intindexlist:
            continue

        with open(args.shname,'a') as f:
            # preprocess
            f.write('conda deactivate'+'\n')
            f.write('conda activate VDW'+'\n')
            f.write('ffmpeg -i ' + todir+'%06d'%(i+1)+'/rgb.mp4 -vf "stereo3d=sbsl:ml,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ' + todir+'%06d'%(i+1)+'/rgbl.mp4'+'\n')
            f.write('ffmpeg -i ' + todir+'%06d'%(i+1)+'/rgb.mp4 -vf "stereo3d=sbsl:mr,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ' + todir+'%06d'%(i+1)+'/rgbr.mp4'+'\n')
            f.write('python ./process/extract_frames.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            if args.cut_black_bar:
                f.write('python ./process/cut_edge.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            f.write('python ./process/readrgb.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            f.write('python ./process/fliprgb.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            f.write('python ./process/lrf2video.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            
            # segformer
            f.write('python ./sky/SegFormer-master/demo/image_demo.py ./sky/SegFormer-master/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py ./sky/SegFormer-master/checkpoints/segformer.b5.640x640.ade.160k.pth --device cuda:'+args.cuda+' --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            
            #mask2former
            f.write('conda deactivate'+'\n')
            f.write('conda activate mask2former'+'\n')
            f.write('CUDA_VISIBLE_DEVICES='+args.cuda+' python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ' + todir+'%06d'%(i+1)+'/leftrgb.avi --base_dir ' + todir+'%06d'%(i+1)+'/l3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl'+'\n')
            f.write('CUDA_VISIBLE_DEVICES='+args.cuda+' python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ' + todir+'%06d'%(i+1)+'/leftrgb_flip.avi --base_dir ' + todir+'%06d'%(i+1)+'/l4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl'+'\n')
            f.write('CUDA_VISIBLE_DEVICES='+args.cuda+' python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ' + todir+'%06d'%(i+1)+'/rightrgb.avi --base_dir ' + todir+'%06d'%(i+1)+'/r3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl'+'\n')
            f.write('CUDA_VISIBLE_DEVICES='+args.cuda+' python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ' + todir+'%06d'%(i+1)+'/rightrgb_flip.avi --base_dir ' + todir+'%06d'%(i+1)+'/r4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl'+'\n')
            
            #sky vote and fill
            f.write('conda deactivate'+'\n')
            f.write('conda activate VDW'+'\n')
            f.write('python ./process/vote_sky.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
            f.write('python ./process/fill_hole.py --base_dir ' + todir+'%06d'%(i+1)+'/'+'\n')
             
            #Gmflow disparity
            f.write('CUDA_VISIBLE_DEVICES='+args.cuda+' python ./gmflow-main/main_gray.py --batch_size 2 --inference_dir ' + todir+'%06d'%(i+1)+'/rgblr/ --dir_paired_data  --output_path ' + todir+'%06d'%(i+1)+'/flow/ --resume ./gmflow-main/pretrained/gmflow_sintel-0c07dcb3.pth --pred_bidir_flow --fwd_bwd_consistency_check --base_dir ' + todir+'%06d'%(i+1)+'/'+' --inference_size 720 1280' + '\n')

    os.system("chmod +x {}".format(args.shname))