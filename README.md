# Deep Dense Multi-Path Neural Network For Prostate Segmentation In Magnetic Resonance Imaging [(Paper)](https://link.springer.com/article/10.1007/s11548-018-1841-4)
This repository contains the code for prostate segmentation in MRI using deep dense multi-path neural network. 

## Task lists
- [x] Preprocessing
- [x] Training and validation
- [ ] Cascade
- [ ] Inference on the test set
- [ ] Resample the test set predictions to their original spacing
- [ ] Usage guideline

## Method
The Preprocessing and Augmentation pipeline are designed specifically for PROMISE12 dataset. However, they can be easily extended for using with other datasets.

The CNN Architecture proposed in our paper:
![alt text](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs11548-018-1841-4/MediaObjects/11548_2018_1841_Fig2_HTML.png?as=webp)

## Private Dataset
Our model achieved **95.11 DSC** on our private dataset when training on both T2 and ADC. 
![abc](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs11548-018-1841-4/MediaObjects/11548_2018_1841_Fig3_HTML.jpg?as=webp "On private dataset")

## PROMISE12
The PROMISE12 dataset contains heterogenous data from 4 different hospitals and institutes. All images were resampled to the spacing of (0.27, 0.27, 2.2) [x, y ,z].
The training set were split into the training* and validation sets based on the [patient meta-data](https://github.com/minhto2802/3d-prostate-segmentation-mxnet/tree/master/src) to ensure that both the training* and validation sets contain data from all 4 hospitals/institues. The model achieved an average ~89 DSC on the validation set without further post-processing.

Below are a few examples of the model predictions on the validation set (yellow: groundtruth, red: prediction):
- **Case00**  
![case00](https://github.com/minhto2802/dense-multipath-nn-prostate-segmentation/blob/master/src/Case00.png)
- **Case15**  
![case15](https://github.com/minhto2802/dense-multipath-nn-prostate-segmentation/blob/master/src/Case15.png)
- **Case25**  
![Case25](https://github.com/minhto2802/dense-multipath-nn-prostate-segmentation/blob/master/src/Case26.png)

## Data Directory
```
.
├── inputs
│   ├── raw
│   │   ├── test
│   │   │   ├── Case00.mhd
│   │   │   ├── Case00.raw
│   │   │   ├── Case00_segmentation.mhd
│   │   │   ├── Case00_segmentation.raw
│   │   │   ├── ...
│   │   │   ├── Case29.mhd
│   │   │   ├── Case29.raw
│   │   │   ├── Case29_segmentation.mhd
│   │   │   ├── Case29_segmentation.raw
│   │   └── training
│   │   │   ├── Case00.mhd
│   │   │   ├── Case00.raw
│   │   │   ├── Case00_segmentation.mhd
│   │   │   ├── Case00_segmentation.raw
│   │   │   ├── ...
│   │   │   ├── Case49.mhd
│   │   │   ├── Case49.raw
│   │   │   ├── Case49_segmentation.mhd
│   │   │   ├── Case49_segmentation.raw
```

## Dependencies
```
gluoncv==0.4.0
imageio==2.8.0
imgaug==0.3.0
matplotlib==3.2.0
mxboard==0.1.0
mxnet-cu102mkl==1.6.0
numpy==1.17.4
opencv-python==4.2.0.32
scikit-image==0.16.2
simpleitk==1.2.4
```

## Authours
* [**Minh Nguyen Nhat To**](https://github.com/minhto2802)

## Citation
If any part of this code is used, please give appropriate citation to our paper.

BibTex entry:  
```
@article{to2018deep,
  title={Deep dense multi-path neural network for prostate segmentation in magnetic resonance imaging},
  author={To, Minh Nguyen Nhat and Vu, Dang Quoc and Turkbey, Baris and Choyke, Peter L and Kwak, Jin Tae},
  journal={International journal of computer assisted radiology and surgery},
  volume={13},
  number={11},
  pages={1687--1696},
  year={2018},
  publisher={Springer}
}
```

**_To Be Updated_** :+1:
