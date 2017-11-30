import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class CapsuleConv2d(nn.Module):
    r"""Applies a 2D capsule convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}

    where :math:`\star` is the valid 2D `cross-correlation`_ operator

    | :attr:`stride` controls the stride for the cross-correlation.
    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded
      on both sides for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])

    Examples::

        >>> import capsulelayer
        >>> from torch.autograd import Variable
        >>> # With square kernels and equal stride
        >>> m = capsulelayer.CapsuleConv2d(16, 33, 3, 4, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m1 = capsulelayer.CapsuleConv2d(16, 33, (3, 5), 4, 3, stride=(2, 1), padding=(4, 2))
        >>> input = Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([20, 33, 24, 49])
        >>> output = m1(input)
        >>> print(output.size())
        torch.Size([20, 33, 28, 100])
    """

    def __init__(self, in_channels, out_channels, kernel_size, in_length, out_length, stride=1,
                 padding=0, num_iterations=3):
        super(CapsuleConv2d, self).__init__()
        if in_channels % in_length != 0:
            raise ValueError('in_channels must be divisible by in_length')
        if out_channels % out_length != 0:
            raise ValueError('out_channels must be divisible by out_length')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_length = in_length
        self.out_length = out_length
        self.stride = stride
        self.padding = padding
        self.num_iterations = num_iterations
        self.weight = Parameter(
            torch.randn(out_channels // out_length, in_channels // in_length, 1,
                        self.kernel_size[0] * self.kernel_size[1],
                        in_length, out_length))

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

        N, C_in, H_in, W_in = input.size()
        H_out = 1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]
        W_out = 1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]
        input_pad = Variable(torch.zeros(N, C_in, H_in + 2 * self.padding[0], W_in + 2 * self.padding[1]))
        out = Variable(torch.zeros((N, self.out_channels, H_out, W_out)))
        if torch.cuda.is_available():
            input_pad = input_pad.cuda()
            out = out.cuda()
        input_pad[:, :, self.padding[0]:self.padding[0] + H_in, self.padding[1]:self.padding[1] + W_in] = input

        input_planes = input_pad.chunk(num_chunks=self.in_channels // self.in_length, dim=1)
        for index in range(self.out_channels // self.out_length):
            for i, plane in enumerate(input_planes):
                for j in range(H_out):
                    for k in range(W_out):
                        window = plane[:, :, j * self.stride[0]:j * self.stride[0] + self.kernel_size[0],
                                 k * self.stride[1]:k * self.stride[1] + self.kernel_size[1]]
                        window = window.contiguous().view(window.size(0), window.size(1), -1).transpose(1, 2)
                        plane_out = route(window, self.weight[index, i], self.num_iterations)
                        out[:, index * self.out_length:(index + 1) * self.out_length, j, k] = \
                            out[:, index * self.out_length:(index + 1) * self.out_length, j, k].add(
                                plane_out.transpose(1, 2))
        return out

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', in_length={in_length}, out_length={out_length}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CapsuleLinear(nn.Module):
    r"""Applies a fully connection capsules to the incoming data

     Args:
         in_capsules: number of each input sample's capsules
         out_capsules: number of each output sample's capsules
         in_length: length of each input sample's each capsule
         out_length: length of each output sample's each capsule
         num_iterations: number of routing iterations

     Shape:
         - Input: :math:`(N, in\_capsules, in\_length)`
         - Output: :math:`(N, out\_capsules, out\_length)`

     Attributes:
         weight: the learnable weights of the module of shape
             (out_capsules x in_capsules x in_length x out_length)

     Examples::
         >>> import capsulelayer
         >>> from torch.autograd import Variable
         >>> m = capsulelayer.CapsuleLinear(20, 30, 8, 16)
         >>> input = Variable(torch.randn(128, 20, 8))
         >>> output = m(input)
         >>> print(output.size())
         torch.Size([128, 30, 16])
     """

    def __init__(self, in_capsules, out_capsules, in_length, out_length, num_iterations=3):
        super(CapsuleLinear, self).__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.num_iterations = num_iterations
        self.weight = Parameter(torch.randn(out_capsules, in_capsules, in_length, out_length))

    def forward(self, input):
        return route(input, self.weight, self.num_iterations)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_capsules) + ' -> ' \
               + str(self.out_capsules) + ')'


def route(input, weight, num_iterations):
    priors = input[None, :, :, None, :] @ weight[:, None, :, :, :]
    logits = Variable(torch.zeros(*priors.size()))
    if torch.cuda.is_available():
        logits = logits.cuda()
    for i in range(num_iterations):
        probs = softmax(logits, dim=2)
        outputs = squash((probs * priors).sum(dim=2, keepdim=True))

        if i != num_iterations - 1:
            delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
            logits = logits + delta_logits
    return outputs.squeeze(dim=2).squeeze(dim=2).transpose(0, 1)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def softmax(tensor, dim=1):
    transposed_input = tensor.transpose(dim, len(tensor.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(tensor.size()) - 1)
