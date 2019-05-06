import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class GridAttentionBlock(nn.Module):
    r"""Applies an grid attention over an input signal
    Reference papers
    Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114
    Reference code
    https://github.com/ozan-oktay/Attention-Gated-Networks
    Args:
        in_features_l (int): Number of channels in the input tensor
        in_features_g (int): Number of channels in the output tensor
        attn_features (int): Number of channels in the middle tensor
    """

    def __init__(self, in_features_l, in_features_g, attn_features):
        super(GridAttentionBlock, self).__init__()
        attn_features = attn_features if attn_features > 0 else 1

        self.W_l = nn.Conv3d(in_features_l, attn_features, kernel_size=1, bias=False)
        self.W_g = nn.Conv3d(in_features_g, attn_features, kernel_size=1, bias=True)
        self.phi = nn.Conv3d(attn_features, 1, kernel_size=1, bias=True)

    def forward(self, l, g):
        if l.size()[2:] != g.size()[2:]:
            l = F.interpolate(l, size=g.size()[2:], mode='trilinear', align_corners=False)
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        c = self.phi(F.relu(l_ + g_))
        # compute attention map
        a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)
        return f


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
        use_attn (bool, optional): If ``True``, use grid attention to the input. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_attn=True):
        super(SpatioTemporalConv, self).__init__()

        self.use_attn = use_attn
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

        if use_attn:
            self.attn = GridAttentionBlock(in_channels, out_channels, in_channels // 2)
            self.conv = nn.Conv3d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=1,
                                  bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu(self.bn1(self.spatial_conv(x)))
        res = self.relu(self.bn2(self.temporal_conv(res)))
        if self.use_attn:
            attend = self.attn(x, res)
            out = self.conv(torch.cat((attend, res), 1))
            return out
        else:
            return res


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
        use_attn (bool, optional): If ``True``, use grid attention to the input. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_attn=True):
        super(TemporalSpatioConv, self).__init__()

        self.use_attn = use_attn
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

        if use_attn:
            self.attn = GridAttentionBlock(in_channels, out_channels, in_channels // 2)
            self.conv = nn.Conv3d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=1,
                                  bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu(self.bn1(self.temporal_conv(x)))
        res = self.relu(self.bn2(self.spatial_conv(res)))
        if self.use_attn:
            attend = self.attn(x, res)
            out = self.conv(torch.cat((attend, res), 1))
            return out
        else:
            return res


class ResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv or TemporalSpatioConv in the standard ResNet
    block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        conv_type (Module, optional): Type of conv that is to be used to form the block. Default: SpatioTemporalConv
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        use_attn (bool, optional): If ``True``, use grid attention to the input. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, conv_type=SpatioTemporalConv, downsample=False,
                 use_attn=True):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride=2
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, stride=2, bias=False,
                                   use_attn=use_attn)
            self.downsampleconv = conv_type(in_channels, out_channels, kernel_size=1, stride=2, bias=False,
                                            use_attn=use_attn)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, bias=False,
                                   use_attn=use_attn)

        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = conv_type(out_channels, out_channels, kernel_size, padding=padding, bias=False, use_attn=use_attn)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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
        use_attn (bool, optional): If ``True``, use grid attention to the input. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalConv,
                 downsample=False, use_attn=True):

        super(ResLayer, self).__init__()

        # implement the first block
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, block_type, downsample, use_attn)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical
            self.blocks += [ResBlock(out_channels, out_channels, kernel_size, block_type, use_attn=use_attn)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class FeatureLayer(nn.Module):
    r"""Forms a feature layer by initializing 5 layers, with the number of blocks in each layer set by layer_sizes,
    and by performing a global average pool at the end producing a 512-dimensional vector for each element in the batch.
    Args:
        layer_sizes (tuple): An iterable containing the number of blocks in each layer
        block_type (Module, optional): Type of block that is to be used to form the block. Default: SpatioTemporalConv
        use_attn (bool, optional): If ``True``, use grid attention to the input. Default: ``True``
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalConv, use_attn=True):
        super(FeatureLayer, self).__init__()

        self.conv1 = block_type(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False, use_attn=False)
        self.conv2 = ResLayer(64, 64, 3, layer_sizes[0], block_type=block_type, use_attn=False)
        self.conv3 = ResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True, use_attn=use_attn)
        self.conv4 = ResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True, use_attn=use_attn)
        self.conv5 = ResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True, use_attn=use_attn)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    r"""Forms a complete two-stream ResNet classifier producing vectors of size num_classes, by initializing a feature
    layers, and passing them through a Linear layer.
    Args:
        num_classes(int): Number of classes in the data
        layer_sizes (tuple): An iterable containing the number of blocks in each layer
        model_type (string): Type of model that is to be used
    """

    def __init__(self, num_classes, layer_sizes, model_type):
        super(Model, self).__init__()

        self.model_type = model_type
        if 'a' in model_type:
            use_attn = True
        else:
            use_attn = False

        if 'st' in model_type:
            # SpatioTemporal Stream
            self.feature_st = FeatureLayer(layer_sizes, block_type=SpatioTemporalConv, use_attn=use_attn)
            self.fc_st = nn.Linear(512, num_classes)

        if 'ts' in model_type:
            # TemporalSpatio Stream
            self.feature_ts = FeatureLayer(layer_sizes, block_type=TemporalSpatioConv, use_attn=use_attn)
            self.fc_ts = nn.Linear(512, num_classes)

        self.__init_weight()

    def forward(self, x):
        if 'st' in self.model_type:
            # SpatioTemporal pipeline
            x_st = self.feature_st(x)
            logits_st = self.fc_st(x_st)

        if 'ts' in self.model_type:
            # TemporalSpatio pipeline
            x_ts = self.feature_ts(x)
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
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
