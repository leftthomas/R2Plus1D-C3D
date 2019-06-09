# C3D
A PyTorch implementation of C3D and R2Plus1D based on CVPR 2014 paper 
[Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) and CVPR 2017
paper [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248).

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
- rar
```
sudo apt install rar
```
- unrar
```
sudo apt install unrar
```
- ffmpeg
```
sudo apt install build-essential openssl libssl-dev autoconf automake cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev nasm yasm libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev
wget https://ffmpeg.org/releases/ffmpeg-4.1.3.tar.bz2
tar -jxvf ffmpeg-4.1.3.tar.bz2
cd ffmpeg-4.1.3/
./configure --prefix="../build" --enable-static --enable-gpl --enable-libass --enable-libfdk-aac --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265 --enable-nonfree --enable-openssl
make -j4
make install
sudo cp ../build/bin/ffmpeg /usr/local/bin/ 
rm -rf ../ffmpeg-4.1.3/ ../ffmpeg-4.1.3.tar.bz2 ../build/
```
- youtube-dl
```
pip install youtube-dl
```
- joblib
```
pip install joblib
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```

## Datasets
The datasets are coming from [UCF101](http://crcv.ucf.edu/data/UCF101.php)、 
[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
and [KINETICS600](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).
Download `UCF101` and `HMDB51` datasets with `train/val/test` split files into `data` directory.
We use the `split1` to split files. Run `misc.py` to preprocess these datasets.

For `KINETICS600` dataset, first download `train/val/test` split files into `data` directory, then 
run `download.py` to download and preprocess this dataset.

## Usage
### Train Model
```
visdom -logging_level WARNING & python train.py --num_epochs 20 --pre_train kinetics600_r2plus1d.pth
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51', 'kinetics600'])
--gpu_ids                     selected gpu [default value is '0,1,2,3']
--model_type                  model type [default value is 'r2plus1d'](choices=['r2plus1d', 'c3d'])
--batch_size                  training batch size [default value is 64]
--num_epochs                  training epochs number [default value is 100]
--pre_train                   used pre-trained model epoch name [default value is None]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser.

### Inference Video
```
python inference.py --video_name data/ucf101/ApplyLipstick/v_ApplyLipstick_g04_c02.avi
optional arguments:
--data_type                   dataset type [default value is 'ucf101'](choices=['ucf101', 'hmdb51', 'kinetics600'])
--model_type                  model type [default value is 'r2plus1d'](choices=['r2plus1d', 'c3d'])
--video_name                  test video name
--model_name                  model epoch name [default value is 'ucf101_r2plus1d.pth']
```
The inferences will show in a pop up window.

## Benchmarks
Adam optimizer (lr=0.0001) were used with learning rate scheduling. 
The model was trained with 100 epochs and batch size of 64 on 4 NVIDIA Tesla V100 (32G) GPUs. 

The videos are preprocessed as 32 frames of 128x128, and cropped to 112x112.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>UCF101</th>
      <th>HMDB51</th>
      <th>Kinetics600</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Num. of Train Videos</td>
      <td align="center">9,537</td>
      <td align="center">756</td>
      <td align="center">4,110</td>
    </tr>
    <tr>
      <td align="center">Num. of Val Videos</td>
      <td align="center">756</td>
      <td align="center">344</td>
      <td align="center">4,110</td>
    </tr>
    <tr>
      <td align="center">Num. of Test Videos</td>
      <td align="center">3,783</td>
      <td align="center">344</td>
      <td align="center">4,110</td>
    </tr>
    <tr>
      <td align="center">Num. of Classes</td>
      <td align="center">101</td>
      <td align="center">2</td>
      <td align="center">2</td>
    </tr>
    <tr>
      <td align="center">R2Plus1D</td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>58.59±2.47</b></td>
      <td align="center"><b>74.44±0.47</b></td>
    </tr>
    <tr>
      <td align="center">C3D</td>
      <td align="center">81.67±9.64</td>
      <td align="center">59.12±11.27</td>
      <td align="center">75.72±3.13</td>
    </tr>
    <tr>
      <td align="center">Num. of Parameters (R2Plus1D)</td>
      <td align="center">33,220,990</td>
      <td align="center">52,387</td>
      <td align="center">52,995</td>
    </tr>
    <tr>
      <td align="center">Num. of Parameters (C3D)</td>
      <td align="center">52,035</td>
      <td align="center">52,387</td>
      <td align="center">52,995</td>
    </tr>
    <tr>
      <td align="center">Training Time (R2Plus1D)</td>
      <td align="center">3min</td>
      <td align="center">6.77s</td>
      <td align="center">61.04s</td>
    </tr>
    <tr>
      <td align="center">Training Time (C3D)</td>
      <td align="center">4.48s</td>
      <td align="center">6.77s</td>
      <td align="center">61.04s</td>
    </tr>
  </tbody>
</table>

## Results
The train/val/test loss、accuracy and confusion matrix are showed on visdom. 
### UCF101
![result](results/ucf101.png)
### HMDB51
![result](results/hmdb51.png)
### KINETICS600
![result](results/kinetics600.png)

