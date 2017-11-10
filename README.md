# ImageClassification
Image Classification

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c soumith
conda install pytorch torchvision cuda80 -c soumith # install it if you have installed cuda
```
* PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* tqdm
```
pip install tqdm
```

## Usage
```
python -m visdom.server & python main.py
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser, or your own host address if specified.
