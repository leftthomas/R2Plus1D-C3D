# Two-Stream ST-TS
A PyTorch implementation of Two-Stream Spatio-Temporal and Temporal-Spatio Convolutional Network based on the paper 
[Two-Stream Spatio-Temporal and Temporal-Spatio Convolutional Network for Activity Recoginition]().

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
- rarfile
```
pip install rarfile
```
- joblib
```
pip install joblib
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
The datasets are coming from [UCF101](http://crcv.ucf.edu/data/UCF101.php)、 
[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
and [KINETICS600](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).
Download `UCF101` and `HMDB51` datasets with `train/val/test` split files into `data` directory.
We use the `split1` to split files. Run `misc.py` to preprocess these datasets.

For `KINETICS600` dataset, first download `train/val/test` split files into `data` directory, and 
then run `download.py` to download and preprocess this dataset.

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51', 'kinetics600'])
--batch_size                  training batch size [default value is 15]
--num_epochs                  training epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

### Inference Video
```
python inference.py --video_name data/ucf101/ApplyLipstick/v_ApplyLipstick_g04_c02.avi
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51', 'kinetics600'])
--video_name                  test video name
--model_name                  model epoch name [default value is 'ucf101_100.pth']
```
The inferences will show in a pop up window.

## Results
The train/val/test loss、accuracy and confusion matrix are showed on visdom. 
### UCF101
![result](results/ucf101.png)
### HMDB51
![result](results/hmdb51.png)
### KINETICS600
![result](results/kinetics600.png)

