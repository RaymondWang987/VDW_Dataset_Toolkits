from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2
import glob
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference') 
    parser.add_argument(
        '--palette',
        default= 'ade',
        help='Color palette used for segmentation map')   #ade
    parser.add_argument('-bd', '--base_dir')
    args = parser.parse_args()

    base_dir = args.base_dir
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    left = glob.glob(base_dir+'/left/*.png')
    left_flip = glob.glob(base_dir+'/left_flip/*.png')
    right = glob.glob(base_dir+'/right/*.png')
    right_flip = glob.glob(base_dir+'/right_flip/*.png')
    for i in range(len(left)):
        print(i)
        img = left[i]
        result = inference_segmentor(model, img)
        show_result_pyplot(model, img, result, get_palette(args.palette),savedir=base_dir+'/left_seg/'+right[i][-16:])
        seqq = result[0]

    
        seqq[seqq!=2] = 0    #2
        seqq[seqq==2] = 255
        
        #print(seqq.shape)
        #plt.imsave(base_dir+'/left_sky/'+left[i][-16:],mmcv.bgr2rgb(seqq))
        cv2.imwrite(base_dir+'/l1/'+left[i][-16:],seqq)

        img = left_flip[i]
        result = inference_segmentor(model, img)
        show_result_pyplot(model, img, result, get_palette(args.palette),savedir=base_dir+'/left_seg/'+right[i][-16:])
        seqq = result[0]

    
        seqq[seqq!=2] = 0    #2
        seqq[seqq==2] = 255
        
        #print(seqq.shape)
        #plt.imsave(base_dir+'/left_sky/'+left[i][-16:],mmcv.bgr2rgb(seqq))
        seqq = cv2.flip(seqq, 1)   ######flip
        cv2.imwrite(base_dir+'/l2/'+left[i][-16:],seqq) 
        
        
        img = right[i]
        result = inference_segmentor(model, img)
        show_result_pyplot(model, img, result, get_palette(args.palette),savedir=base_dir+'/right_seg/'+right[i][-16:])
        seqq = result[0]
    
        seqq[seqq!=2] = 0     #2
        seqq[seqq==2] = 255
        
        #print(seqq.shape)
        #plt.imsave(base_dir+'/right_sky/'+right[i][-16:],mmcv.bgr2rgb(seqq))
        cv2.imwrite(base_dir+'/r1/'+right[i][-16:],seqq)

        img = right_flip[i]
        result = inference_segmentor(model, img)
        show_result_pyplot(model, img, result, get_palette(args.palette),savedir=base_dir+'/right_seg/'+right[i][-16:])
        seqq = result[0]
    
        seqq[seqq!=2] = 0     #2
        seqq[seqq==2] = 255
        
        #print(seqq.shape)
        #plt.imsave(base_dir+'/right_sky/'+right[i][-16:],mmcv.bgr2rgb(seqq))
        seqq = cv2.flip(seqq, 1)   ######flip
        cv2.imwrite(base_dir+'/r2/'+right[i][-16:],seqq)

    #show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
