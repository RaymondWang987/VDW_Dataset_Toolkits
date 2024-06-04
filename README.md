# VDW_Dataset_Toolkits 🚀🚀🚀

🎉🎉🎉 **Welcome to the VDW Dataset Toolkits!** 🎉🎉🎉  

**This repo contains the official generation code of Video Depth in the Wild (VDW) dataset.**

The VDW dataset is proposed by ICCV 2023 paper "Neural Video Depth Stabilizer" ([NVDS repo](https://github.com/RaymondWang987/NVDS))

Authors: [Yiran Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=p_RnaI8AAAAJ)<sup>1</sup>,
[Min Shi](https://www.semanticscholar.org/author/Min-Shi/1516268415)<sup>1</sup>,
[Jiaqi Li](https://scholar.google.com/citations?hl=zh-CN&user=i-2ghuYAAAAJ)<sup>1</sup>,
[Zihao Huang](https://orcid.org/0000-0002-8804-191X)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Jianming Zhang](https://jimmie33.github.io/)<sup>2</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>3*</sup>, 
[Guosheng Lin](https://guosheng.github.io/)<sup>3</sup>


Institutes: <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Adobe Research, <sup>3</sup>Nanyang Technological University

## 🌼 VDW Dataset
<p align="center">
<img src="Assets/vdw1.png" width="90%">
</p>

+ **Dataset Description.** Previous video depth datasets are limited in both diversity and volume. To compensate for the data shortage and boost the performance of learning-based video depth models, we elaborate a large-scale natural-scene dataset, Video Depth in the Wild (VDW). To the best of our knowledge, our VDW dataset is currently the largest video depth dataset with the most diverse video scenes. We collect stereo videos from diverse data sources. The VDW test set is with 90 videos and 12622 frames, while the VDW training set contains 14203 videos with over 2 million frames (8TB on hard drive). We also provide a VDW demo set with two sequences. Users could leverage the VDW official toolkits and demo sequences to learn about our data processing pipeline.

 + **Releasing Policy.** We have released the VDW dataset under strict conditions. We must ensure that the releasing won’t violate any copyright requirements. **To this end, we will not release any video frames or the derived data in public.** Instead, we provide meta data and detailed toolkits, which can be used to reproduce VDW or generate your own data. All the meta data and toolkits are licensed under [CC BY-NC-SA 4.0](https://github.com/RaymondWang987/VDW_Dataset_Toolkits/blob/main/LICENSE), which can only be used for academic and research purposes. Refer to the [VDW official website](https://raymondwang987.github.io/VDW/) for more information.

## 🔨 Installation
+ **Environments.** Two conda envs are required: `VDW` and `mask2former`. The VDW env is based on `python=3.6.13` and `pytorch==1.7.1`. Refer to the `requirements.txt` (directly required packages retrieved by pipreqs) and the `all_package.txt` (all packages retrieved by pip freeze). **We install the basic packages,  [GMFlow](https://github.com/haofeixu/gmflow), and [SegFormer](https://github.com/NVlabs/SegFormer) in the VDW env, while create another mask2former env for [Mask2Former](https://github.com/facebookresearch/Mask2Former).**
  ```
  conda create -n VDW python=3.6.13
  conda activate VDW
  conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.1 -c pytorch -c conda-forge
  pip install numpy imageio opencv-python scipy tensorboard timm scikit-image tqdm glob h5py
  ```
  ```
  # Refer to the installation of Mask2Former.
  conda create -n mask2former python=3.8.13
  conda activate mask2former
  conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
  pip install numpy imageio opencv-python scipy tensorboard timm scikit-image tqdm glob h5py
  ```
+ **Installation of GMflow, Mask2Former, and SegFormer.** We utilize state-of-the-art optical flow model [GMFlow](https://arxiv.org/abs/2111.13680) to generate disparity. The semantic segmentation models Mask2Former and SegFormer are utilized to conduct sky segmentation (infinitely far, i.e., zero disparity). Please refer to [GMFlow](https://github.com/haofeixu/gmflow), [SegFormer](https://github.com/NVlabs/SegFormer) (the two in VDW env), and [Mask2Former](https://github.com/facebookresearch/Mask2Former) (in mask2former env) for installation. 

+ **MMCV and MMseg.** The [SegFormer](https://github.com/NVlabs/SegFormer) also relies on MMSegmentation and MMCV. **we suggest you to install `mmcv-full==1.x.x`**, because some API or functions are removed in `mmcv-full==2.x.x`. Please refer to [MMSegmentation-v0.11.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0) and their [official document](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) for detailed installation instructions step by step. **The key is to match the version of mmcv-full and mmsegmentation with the version of cuda and pytorch on your server.** For instance, I have `CUDA 11.1` and `PyTorch 1.9.0` on my server, thus `mmcv-full 1.3.x` and `mmseg 0.11.0` (as in our installation instructions) are compatible with my environment (confirmed by [mmcv-full 1.3.x](https://mmcv.readthedocs.io/zh_CN/v1.3.13/get_started/installation.html)). You should check the matching version of your own server on the official documents of [mmcv-full](https://mmcv.readthedocs.io/en/latest/) and [mmseg](https://mmsegmentation.readthedocs.io/en/latest/). You can choose different versions in their documents and check the version matching relations. By reading and following the detailed mmcv-full and mmseg documents, the installation seems to be easy. You can also refer to [NVDS Issue #1](https://github.com/RaymondWang987/NVDS/issues/1) and the [SegFormer](https://github.com/NVlabs/SegFormer) for some discussions.

  
## ⚡ Data Generation with VDW Demo Set

## 🍭 Acknowledgement
We thank the authors for releasing [PyTorch](https://pytorch.org/), [MiDaS](https://github.com/intel-isl/MiDaS), [DPT](https://github.com/isl-org/DPT), [GMFlow](https://github.com/haofeixu/gmflow), [SegFormer](https://github.com/NVlabs/SegFormer), [VSS-CFFM](https://github.com/GuoleiSun/VSS-CFFM), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), and [FFmpeg](http://ffmpeg.org/). Thanks for their solid contributions and cheers to the community.

## 📧 Citation
```
@article{wang2023neural,
  title={Neural Video Depth Stabilizer},
  author={Wang, Yiran and Shi, Min and Li, Jiaqi and Huang, Zihao and Cao, Zhiguo and Zhang, Jianming and Xian, Ke and Lin, Guosheng},
  journal={arXiv preprint arXiv:2307.08695},
  year={2023}
}
```
  


