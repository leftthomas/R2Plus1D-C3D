# CCN
A PyTorch implementation of Convolutional Capsule Network based on the paper [Convolutional Capsule Network for Activity Recoginition]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
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
--batch_size                  train batch size [default value is 64]
--num_epochs                  train epochs number [default value is 80]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

## Results
The train loss、accuracy, test loss、accuracy are showed on visdom.
![result](results/mutag.png)
![result](results/ptc.png)

