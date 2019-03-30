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
There are many public datasets for human activity recognition. You can refer to this survey article [Deep learning for sensor-based activity recognition: a survey](https://arxiv.org/abs/1707.03502) to find more.

In this demo, we will use UCI HAR dataset as an example. This dataset can be found in [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/).

Of course, this dataset needs further preprocessing before being put into the network. I've also provided a preprocessing version of the dataset as a `.npz` file so you can focus on the network (download [HERE](https://pan.baidu.com/s/1Nx7UcPqmXVQgNVZv4Ec1yg)). It is also highly recommended you download the dataset so that you can experience all the process on your own.

| #subject | #activity | Frequency |
| --- | --- | --- |
| 30 | 6 | 50 Hz |

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
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, `$data_type` means the dataset 
type which you are training.

### About the inputs
That dataset contains 9 channels of the inputs: (acc_body, acc_total and acc_gyro) on x-y-z. So the input channel is 9.

Dataset providers have clipped the dataset using sliding window, so every 128 in `.txt` can be considered as an input. 
In real life, you need to first clipped the input using sliding window.

So in the end, we reformatted the inputs from 9 inputs files to 1 file, the shape of that file is `[n_sample,128,9]`, 
that is, every windows has 9 channels with each channel has length 128. When feeding it to Tensorflow, it has to be 
reshaped to `[n_sample,9,1,128]` as we expect there is 128 X 1 signals for every channel.

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

