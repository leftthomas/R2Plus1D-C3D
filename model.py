import math

import torch.nn as nn
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


class ResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv or TemporalSpatioConv in the standard ResNet
    block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        conv_type (Module, optional): Type of conv that is to be used to form the block. Default: SpatioTemporalConv
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, conv_type=SpatioTemporalConv, downsample=False):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride=2
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, stride=2)
            self.downsampleconv = conv_type(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = conv_type(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class ResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the layer.
        kernel_size (int or tuple): Size of the convolving kernels.
        layer_size (int): Number of blocks to be stacked to form the layer
        block_type (Module, optional): Type of block that is to be used to form the block. Default: SpatioTemporalConv
        downsample (bool, optional): If ``True``, the first block in the layer will implement downsampling. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalConv,
                 downsample=False):

        super(ResLayer, self).__init__()

        # implement the first block
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, block_type, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical
            self.blocks += [ResBlock(out_channels, out_channels, kernel_size, block_type)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class Model(nn.Module):
    r"""Forms a complete two-stream ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch, and passing them through
    a Linear layer.
    Args:
        num_classes(int): Number of classes in the data
        layer_sizes (tuple): An iterable containing the number of blocks in each layer
        model_type (string): Type of model that is to be used
    """

    def __init__(self, num_classes, layer_sizes, model_type):
        super(Model, self).__init__()

        self.model_type = model_type

        if 'st' in model_type:
            # SpatioTemporal Stream
            self.conv1_st = SpatioTemporalConv(3, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            self.conv2_st = ResLayer(32, 32, 3, layer_sizes[0], block_type=SpatioTemporalConv)
            self.conv3_st = ResLayer(32, 64, 3, layer_sizes[1], block_type=SpatioTemporalConv, downsample=True)
            self.conv4_st = ResLayer(64, 128, 3, layer_sizes[2], block_type=SpatioTemporalConv, downsample=True)
            self.conv5_st = ResLayer(128, 256, 3, layer_sizes[3], block_type=SpatioTemporalConv, downsample=True)
            self.pool_st = nn.AdaptiveAvgPool3d(1)
            self.fc_st = nn.Linear(256, num_classes)

        if 'ts' in model_type:
            # TemporalSpatio Stream
            self.conv1_ts = TemporalSpatioConv(3, 32, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            self.conv2_ts = ResLayer(32, 32, 3, layer_sizes[0], block_type=TemporalSpatioConv)
            self.conv3_ts = ResLayer(32, 64, 3, layer_sizes[1], block_type=TemporalSpatioConv, downsample=True)
            self.conv4_ts = ResLayer(64, 128, 3, layer_sizes[2], block_type=TemporalSpatioConv, downsample=True)
            self.conv5_ts = ResLayer(128, 256, 3, layer_sizes[3], block_type=TemporalSpatioConv, downsample=True)
            self.pool_ts = nn.AdaptiveAvgPool3d(1)
            self.fc_ts = nn.Linear(256, num_classes)

        self.__init_weight()

    def forward(self, x):
        if 'st' in self.model_type:
            # SpatioTemporal pipeline
            x_st = self.conv1_st(x)
            x_st = self.conv2_st(x_st)
            x_st = self.conv3_st(x_st)
            x_st = self.conv4_st(x_st)
            x_st = self.conv5_st(x_st)
            x_st = self.pool_st(x_st)
            x_st = x_st.view(-1, 256)
            logits_st = self.fc_st(x_st)

        if 'ts' in self.model_type:
            # TemporalSpatio pipeline
            x_ts = self.conv1_ts(x)
            x_ts = self.conv2_ts(x_ts)
            x_ts = self.conv3_ts(x_ts)
            x_ts = self.conv4_ts(x_ts)
            x_ts = self.conv5_ts(x_ts)
            x_ts = self.pool_ts(x_ts)
            x_ts = x_ts.view(-1, 256)
            logits_ts = self.fc_ts(x_ts)

        if 'st' in self.model_type and 'ts' in self.model_type:
            logits = (logits_st + logits_ts) / 2
        else:
            if 'st' in self.model_type:
                logits = logits_st
            elif 'ts' in self.model_type:
                logits = logits_ts
            else:
                raise NotImplementedError('check the model type')

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
