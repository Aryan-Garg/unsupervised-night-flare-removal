# SPCULR (CVPR 2024)

## Introduction
This is an implementation of the following paper.
> [Single Photon Camera Guided Unsupervised Low-Light HDR Image Restoration]()\
> Computer Vision and Pattern Recognition (`CVPR2022`)

[Aryan Garg](https://github.com/Aryan-Garg) and [Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra/)

[[Paper]]()
[[Supplementary]]()
[![arXiv]()
[[Poster]]() 
[[Slides]]() 
[[Link]]()

### Abstract


## Datasets
### Light-Effects Suppression on Night Data
1. [Light-effects data](https://www.dropbox.com/sh/ro8fs629ldebzc2/AAD_W78jDffsJhH-smJr0cNSa?dl=0) <br>
Light-effects data is collected from Flickr and by ourselves, with multiple light colors in various scenes: <br>
* `CVPR2021`
*Nighttime Visibility Enhancement by Increasing the Dynamic Range and Suppression of Light Effects* [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sharma_Nighttime_Visibility_Enhancement_by_Increasing_the_Dynamic_Range_and_Suppression_CVPR_2021_paper.pdf)]\
[Aashish Sharma](https://aasharma90.github.io/) and [Robby T. Tan](https://tanrobby.github.io/pub.html)

<p align="left">
  <img width=950" src="teaser/self-collected.png">
</p>

2. [LED data](https://www.dropbox.com/sh/7lhpnj2onb8c3dl/AAC-UF1fvJLxvCG-IuYLQ8T4a?dl=0) <br>
We captured images with dimmer light as the reference images.
<p align="left">
  <img width=350" src="teaser/LED.PNG">
</p>


3. [GTA5 nighttime fog](https://www.dropbox.com/sh/gfw44ttcu5czrbg/AACr2GZWvAdwYPV0wgs7s00xa?dl=0) <br>
Synthetic GTA5 nighttime fog data:<br> 
* `ECCV2020`
*Nighttime Defogging Using High-Low Frequency Decomposition and Grayscale-Color Networks* [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570460.pdf)]\
Wending Yan, [Robby T. Tan](https://tanrobby.github.io/pub.html) and [Dengxin Dai](https://vas.mpi-inf.mpg.de/) 

<p align="left">
  <img width=350" src="teaser/GTA5.PNG">
</p>

4. [Syn-light-effects](https://www.dropbox.com/sh/2sb9na4ur7ry2gf/AAB1-DNxy4Hq6qPU-afYIKVaa?dl=0) <br>
Synthetic-light-effects data is the implementation of the paper, <br>
S. Metari, F. Deschênes, "A New Convolution Kernel for Atmospheric Point Spread Function Applied to Computer Vision", ICCV, 2017. <br>
Run the Matlab code to generate Syn-light-effects:
```
glow_rendering_code/repro_ICCV2007_Fig5.m
```
<p align="left">
  <img width=350" src="teaser/syn.PNG">
</p>

# Light-Effects Suppression Results:
## Pre-trained Model
[Update] We have released light-effects suppression code and checkpoint on May 21, 2023. 
1. Download the [pre-trained de-light-effects model](https://www.dropbox.com/s/9fif8itsu06quvn/delighteffects_params_0600000.pt?dl=0), put in ./results/delighteffects/model/
2. Put the test images in ./light-effects/

## Light-effects Suppression Test
```
python main_delighteffects.py
```

## Demo
[Update] We have released demo_all.html and demo_all.ipynb code on May 21, 2023. 

Input are in ./light-effects/, Output are in ./light-effects-output/
```
demo_all.ipynb
```
<p align="left">
  <img width="950" src="teaser/light_effects.PNG">
</p>



[Update] We have released demo code on Dec 28, 2022.
```
python demo.py
```

## Decomposition
[Update] We have released decomposition code on Dec 28, 2022. 
run the code to layer decomposition, output light-effects layer, initial background layer.    
```
demo_decomposition.m
```
[Background Results](https://www.dropbox.com/sh/bis4350df85gz0e/AAC7wY92U9K5JW3aSaD0mvcya?dl=0) | 
[Light-Effects Results](https://www.dropbox.com/sh/d7myjujl9gwotkz/AAA0iSsO1FbWqNkbB6QR-sLCa?dl=0) |
[Shading Results](https://www.dropbox.com/sh/venya8tvetyiv07/AABud1xlWGVquKptBsIZ0jxpa?dl=0)

<p align="left">
  <img width="550" src="teaser/decomposition.png">
</p>

### Feature Results:
1. run the MATLAB code to adaptively fuse the three color channels, output I_gray
```
checkGrayMerge.m
```
<p align="left">
  <img width="350" src="VGG_code/results_VGGfeatures/DSC01607_I_GrayBest.png">
</p>

2. Download the [fine-tuned VGG model](https://www.dropbox.com/s/xzzoruz1i6m7mm0/model_best.tar?dl=0) (fine-tuned on [ExDark (Exclusively Dark Image Dataset)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)), put in 
./VGG_code/ckpts/vgg16_featureextractFalse_ExDark/nets/model_best.tar

3. obtain structure features
```
python test_VGGfeatures.py
```

## Summary of Comparisons:
<p align="left">
  <img width="550" src="teaser/comparison.png">
</p>


### Low-Light Enhancement
1. [LOL dataset](https://daooshee.github.io/BMVC2018website/) <br>
LOL: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement", BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing) <br>

2. [LOL-Real dataset](https://github.com/flyywh/CVPR-2020-Semi-Low-Light/) <br>
LOL-real (the extension work): Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [[Google Drive]](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) <br> <br>
We use LOL-real as it is larger and more diverse.


# Low-Light Enhancement Results:
## Pre-trained Model

1. Download the [pre-trained LOL model](https://www.dropbox.com/s/0ykpsm1d48f74ao/LOL_params_0900000.pt?dl=0), put in ./results/LOL/model/
2. Put the test images in ./LOL/ 

## Low-light Enhancement Test
```
python main.py
```

## Results
<p align="left">
  <img width="750" src="teaser/lowlight.PNG">
</p>

1. [LOL-Real Results](https://www.dropbox.com/sh/t6eb4aq025ctnhy/AADRRJNN3u-N8HApe1tFo19Ra?dl=0)<br>

Get the following Table 4 in the main paper on the LOL-Real dataset (100 test images).

|Learning| Method | PSNR | SSIM | 
|--------|--------|------|------ |
| Unsupervised Learning| **Ours** | **25.51** |**0.8015**|
| N/A | Input | 9.72 | 0.1752|

<p align="left">
  <img width="550" src="teaser/LOL_real.PNG">
</p>

[Update]: Re-train (train from scratch) in LOL_V2_real (698 train images）, and test on [LOL_V2_real](https://www.dropbox.com/sh/7t1qgl4anlqcvle/AAAyOUHMoG5IkzCX5GQDPd1Oa?dl=0) (100 test images).<br>
PSNR: 20.85 (vs EnlightenGAN's 18.23), SSIM: 0.7243 (vs EnlightenGAN's 0.61)
[pre-trained LOL_V2 model](https://www.dropbox.com/sh/7t1qgl4anlqcvle/AAAyOUHMoG5IkzCX5GQDPd1Oa?dl=0)

2. [LOL-test Results](https://www.dropbox.com/sh/la21ocjk14dtg9t/AABOBsCQ39Oml33fItqX5koFa?dl=0)<br>

Get the following Table 3 in the main paper on the LOL-test dataset (15 test images).
|Learning| Method | PSNR | SSIM | 
|--------|--------|------|------ |
| Unsupervised Learning| **Ours** | **21.521** |**0.7647**|
| N/A | Input | 7.773 | 0.1259|

<p align="left">
  <img width="450" src="teaser/LOL.PNG">
</p>


### Citations
If this work is useful for your research, please cite our paper. 
```BibTeX
To come
```

If light-effects data is useful for your research, please cite their paper. 
```BibTeX
@inproceedings{sharma2021nighttime,
	title={Nighttime Visibility Enhancement by Increasing the Dynamic Range and Suppression of Light Effects},
	author={Sharma, Aashish and Tan, Robby T},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	pages={11977--11986},
	year={2021}
}
```

If GTA5 nighttime fog data is useful for your research, please cite their paper. 
```BibTeX
@inproceedings{yan2020nighttime,
	title={Nighttime defogging using high-low frequency decomposition and grayscale-color networks},
	author={Yan, Wending and Tan, Robby T and Dai, Dengxin},
	booktitle={European Conference on Computer Vision},
	pages={473--488},
	year={2020},
	organization={Springer}
}
```
