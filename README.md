# CCN
A PyTorch implementation of Convolutional Capsule Network based on the paper [Convolutional Capsule Network for Activity Recoginition]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets
TODO

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51'])
--clip_len                    number of frames in each video [default value is 16]
--crop_size                   crop size of video [default value is 112]
--batch_size                  training batch size [default value is 20]
--num_epochs                  training epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

## Results
The train/val/test loss、accuracy and confusion matrix are showed on visdom.
![result](results/mutag.png)
![result](results/ptc.png)

