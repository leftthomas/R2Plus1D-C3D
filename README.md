# CapsGCNN
A PyTorch implementation of Capsule Graph Convolutional Neural Network based on the paper 
[Capsule Graph Convolutional Neural Network For Graph Classification]().

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* PyTorch Geometric
```
pip install torch-geometric
```
* capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets

The datasets are collected from [graph kernel datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets#file_format).
You can download them from there and extract them into `data` directory, or the code will download them automatically.

## Usage
### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py --data_type PTC_MR --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'DD'](choices:['REDDIT-BINARY', 'DD', 'REDDIT-MULTI-12K', 'REDDIT-MULTI-5K', 'PTC_MR', 'NCI1', 'NCI109', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'ENZYMES', 'COLLAB'])
--num_iterations              routing iterations number [default value is 3]
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, `$data_type` means the dataset type which you are training.
