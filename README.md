# CapsGCNN
A PyTorch implementation of Capsule Graph Convolutional Neural Network based on the paper 
[Capsule Graph Convolutional Neural Network For Graph Classification]().

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
- librosa
```
pip install librosa
```
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets
The datasets are collected from [graph kernel datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).
The code will download and extract them into `data` directory automatically. The `10fold_idx` files are collected from 
[pytorch_DGCNN](https://github.com/muhanzhang/pytorch_DGCNN).

## Usage
### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py --data_type AudioMNIST --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'AudioMNIST'](choices:['AudioMNIST', 'UrbanSound8K'])
--num_iterations              routing iterations number [default value is 3]
--batch_size                  train batch size [default value is 20]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, `$data_type` means the dataset type which you are training.

## Benchmarks
Default PyTorch Adam optimizer hyper-parameters were used without learning rate scheduling. 
The model was trained with 100 epochs and batch size of 20 on a NVIDIA GTX 1070 GPU. 

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>MUTAG</th>
      <th>PTC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Num. of Graphs</td>
      <td align="center">188</td>
      <td align="center">344</td>
    </tr>
    <tr>
      <td align="center">Num. of Classes</td>
      <td align="center">2</td>
      <td align="center">2</td>
    </tr>
    <tr>
      <td align="center">Node Attr. (Dim.)</td>
      <td align="center">8</td>
      <td align="center">19</td>
    </tr>
    <tr>
      <td align="center">Num. of Parameters</td>
      <td align="center">52,035</td>
      <td align="center">52,387</td>
    </tr>
    <tr>
      <td align="center">DGCNN</td>
      <td align="center">85.83±1.66</td>
      <td align="center">58.59±2.47</td>
    </tr>    
    <tr>
      <td align="center">Ours</td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>58.59±2.47</b></td>
    </tr>
    <tr>
      <td align="center">Training Time</td>
      <td align="center">4.51s</td>
      <td align="center">6.88s</td>
    </tr> 
  </tbody>
</table>

## Results
The train loss、accuracy, test loss、accuracy are showed on visdom.

### MUTAG
![result](results/mutag.png)
### PTC
![result](results/ptc.png)

