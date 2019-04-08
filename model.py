import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during
              their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over.
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        # compute the number of intermediary channels (M)
        if bias is True:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        1 + kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(intermed_channels)

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


class TemporalSpatioConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 1D convolution over the time
    axis to an intermediate subspace, followed by a 2D convolution over the spatial axes to
    produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during
              their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(TemporalSpatioConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        # decomposing the parameters into temporal and spatial components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over.
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        # compute the number of intermediary channels (M)
        if bias is True:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        1 + kernel_size[1] * kernel_size[2] * out_channels + kernel_size[0] * in_channels)))
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * out_channels + kernel_size[0] * in_channels)))

        self.temporal_conv = nn.Conv3d(in_channels, intermed_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(intermed_channels)

        self.spatial_conv = nn.Conv3d(intermed_channels, out_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.temporal_conv(x)))
        x = self.relu(self.bn2(self.spatial_conv(x)))
        return x


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        # SpatioTemporal Stream
        self.conv1a = SpatioTemporalConv(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.pool1a = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = SpatioTemporalConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.pool2a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = SpatioTemporalConv(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.conv3aa = SpatioTemporalConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3aa = nn.BatchNorm3d(num_features=256)
        self.pool3a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = SpatioTemporalConv(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(num_features=512)
        self.conv4aa = SpatioTemporalConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4aa = nn.BatchNorm3d(num_features=512)
        self.pool4a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = SpatioTemporalConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(num_features=512)
        self.conv5aa = SpatioTemporalConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5aa = nn.BatchNorm3d(num_features=512)
        self.pool5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6a = nn.Linear(8192, 4096)
        self.fc7a = nn.Linear(4096, 2048)
        self.fc8a = nn.Linear(2048, num_classes)

        # TemporalSpatio Stream
        self.conv1b = TemporalSpatioConv(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.pool1b = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2b = TemporalSpatioConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.pool2b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3b = TemporalSpatioConv(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.conv3bb = TemporalSpatioConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3bb = nn.BatchNorm3d(num_features=256)
        self.pool3b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4b = TemporalSpatioConv(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(num_features=512)
        self.conv4bb = TemporalSpatioConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4bb = nn.BatchNorm3d(num_features=512)
        self.pool4b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5b = TemporalSpatioConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(num_features=512)
        self.conv5bb = TemporalSpatioConv(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5bb = nn.BatchNorm3d(num_features=512)
        self.pool5b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6b = nn.Linear(8192, 4096)
        self.fc7b = nn.Linear(4096, 2048)
        self.fc8b = nn.Linear(2048, num_classes)

        # common modules
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # SpatioTemporal pipeline
        x_a = self.relu(self.bn1a(self.conv1a(x)))
        x_a = self.pool1a(x_a)

        x_a = self.relu(self.bn2a(self.conv2a(x_a)))
        x_a = self.pool2a(x_a)

        x_a = self.relu(self.bn3a(self.conv3a(x_a)))
        x_a = self.relu(self.bn3aa(self.conv3aa(x_a)))
        x_a = self.pool3a(x_a)

        x_a = self.relu(self.bn4a(self.conv4a(x_a)))
        x_a = self.relu(self.bn4aa(self.conv4aa(x_a)))
        x_a = self.pool4a(x_a)

        x_a = self.relu(self.bn5a(self.conv5a(x_a)))
        x_a = self.relu(self.bn5aa(self.conv5aa(x_a)))
        x_a = self.pool5a(x_a)

        x_a = x_a.view(-1, 8192)
        x_a = self.relu(self.fc6a(x_a))
        x_a = self.dropout(x_a)
        x_a = self.relu(self.fc7a(x_a))
        x_a = self.dropout(x_a)
        logits_a = self.fc8a(x_a)

        # TemporalSpatio pipeline
        x_b = self.relu(self.bn1b(self.conv1b(x)))
        x_b = self.pool1b(x_b)

        x_b = self.relu(self.bn2b(self.conv2b(x_b)))
        x_b = self.pool2b(x_b)

        x_b = self.relu(self.bn3b(self.conv3b(x_b)))
        x_b = self.relu(self.bn3bb(self.conv3bb(x_b)))
        x_b = self.pool3b(x_b)

        x_b = self.relu(self.bn4b(self.conv4b(x_b)))
        x_b = self.relu(self.bn4bb(self.conv4bb(x_b)))
        x_b = self.pool4b(x_b)

        x_b = self.relu(self.bn5b(self.conv5b(x_b)))
        x_b = self.relu(self.bn5bb(self.conv5bb(x_b)))
        x_b = self.pool5b(x_b)

        x_b = x_b.view(-1, 8192)
        x_b = self.relu(self.fc6b(x_b))
        x_b = self.dropout(x_b)
        x_b = self.relu(self.fc7b(x_b))
        x_b = self.dropout(x_b)
        logits_b = self.fc8b(x_b)

        logits = (F.softmax(logits_a, dim=-1) + F.softmax(logits_b, dim=-1)) / 2

        return logits
