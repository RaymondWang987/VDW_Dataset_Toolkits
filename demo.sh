# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/wangyiran/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/wangyiran/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/wangyiran/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/wangyiran/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda deactivate
conda activate VDW
ffmpeg -i ./VDW_Demo_Dataset/processed_dataset/000001/rgb.mp4 -vf "stereo3d=sbsl:ml,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ./VDW_Demo_Dataset/processed_dataset/000001/rgbl.mp4
ffmpeg -i ./VDW_Demo_Dataset/processed_dataset/000001/rgb.mp4 -vf "stereo3d=sbsl:mr,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ./VDW_Demo_Dataset/processed_dataset/000001/rgbr.mp4
python ./process/extract_frames.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
python ./process/readrgb.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
python ./process/fliprgb.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
python ./process/lrf2video.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
python ./sky/SegFormer-master/demo/image_demo.py ./sky/SegFormer-master/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py ./sky/SegFormer-master/checkpoints/segformer.b5.640x640.ade.160k.pth --device cuda:0 --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
conda deactivate
conda activate mask2former
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000001/leftrgb.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/l3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000001/leftrgb_flip.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/l4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000001/rightrgb.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/r3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000001/rightrgb_flip.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/r4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
conda deactivate
conda activate VDW
python ./process/vote_sky.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
python ./process/fill_hole.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/
CUDA_VISIBLE_DEVICES=0 python ./gmflow-main/main_gray.py --batch_size 2 --inference_dir ./VDW_Demo_Dataset/processed_dataset/000001/rgblr/ --dir_paired_data  --output_path ./VDW_Demo_Dataset/processed_dataset/000001/flow/ --resume ./gmflow-main/pretrained/gmflow_sintel-0c07dcb3.pth --pred_bidir_flow --fwd_bwd_consistency_check --base_dir ./VDW_Demo_Dataset/processed_dataset/000001/ --inference_size 720 1280
conda deactivate
conda activate VDW
ffmpeg -i ./VDW_Demo_Dataset/processed_dataset/000002/rgb.mp4 -vf "stereo3d=sbsl:ml,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ./VDW_Demo_Dataset/processed_dataset/000002/rgbl.mp4
ffmpeg -i ./VDW_Demo_Dataset/processed_dataset/000002/rgb.mp4 -vf "stereo3d=sbsl:mr,scale=iw*2:ih" -x264-params "crf=24" -c:a copy -y ./VDW_Demo_Dataset/processed_dataset/000002/rgbr.mp4
python ./process/extract_frames.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
python ./process/readrgb.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
python ./process/fliprgb.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
python ./process/lrf2video.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
python ./sky/SegFormer-master/demo/image_demo.py ./sky/SegFormer-master/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py ./sky/SegFormer-master/checkpoints/segformer.b5.640x640.ade.160k.pth --device cuda:0 --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
conda deactivate
conda activate mask2former
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000002/leftrgb.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/l3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000002/leftrgb_flip.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/l4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000002/rightrgb.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/r3/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
CUDA_VISIBLE_DEVICES=0 python ./sky/Mask2Former/demo/demo.py --config-file ./sky/Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml --video-input ./VDW_Demo_Dataset/processed_dataset/000002/rightrgb_flip.avi --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/r4/ --mode noflip --opts MODEL.WEIGHTS ./sky/Mask2Former/checkpoints/model_final_6b4a3a.pkl
conda deactivate
conda activate VDW
python ./process/vote_sky.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
python ./process/fill_hole.py --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/
CUDA_VISIBLE_DEVICES=0 python ./gmflow-main/main_gray.py --batch_size 2 --inference_dir ./VDW_Demo_Dataset/processed_dataset/000002/rgblr/ --dir_paired_data  --output_path ./VDW_Demo_Dataset/processed_dataset/000002/flow/ --resume ./gmflow-main/pretrained/gmflow_sintel-0c07dcb3.pth --pred_bidir_flow --fwd_bwd_consistency_check --base_dir ./VDW_Demo_Dataset/processed_dataset/000002/ --inference_size 720 1280
