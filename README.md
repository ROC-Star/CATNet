# CATNet: A Cascaded and Aggregated Transformer Network For RGB-D Salient Object Detection
The paper has been online published by IEEE Transactions Multimedia.
![](./figs/Overview.png)

## Requirements
python 3.9

pytorch 1.11.0

tensorboardX 2.5
## Results and Saliency maps
We provide saliency maps([Google Drive](https://drive.google.com/drive/folders/1DZGmNBl3jZBGexlosbWcrhQYQtWdE2Bv?usp=drive_link/)) of our CATNet on 7 datasets.
## Training
Please run 
```
CatNet_train.py
```
## Pre-trained model and testing
- Download the following pre-trained models ([Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)) and put them in /pre.
- Modify pathes of pre-trained models and datasets.
- Run 
```
CatNet_test.py
```
Datasets:
[Google Drive](https://drive.google.com/file/d/1ZF94G7ZwRo7M5M2qegvY_BYbwdN17mwk/view?usp=drive_link)


If you anywhere questions, please tell me([724162106@qq.com]).
