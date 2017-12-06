from torch.autograd import Function


def capsuleconv2d(input, weight, stride, padding, ):
    if input is not None and input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_pair(stride), _pair(padding), _pair(dilation), False,
                _pair(0), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


class CapsuleConv2d(Function):
    def __init__(self, weight, in_channels, kernel_size, in_length, out_length, stride,
                 padding, num_iterations):
        super(CapsuleConv2d, self).__init__()
        self.weight = weight
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.in_length = in_length
        self.out_length = out_length
        self.stride = stride
        self.padding = padding
        self.num_iterations = num_iterations

    @staticmethod
    def forward(self, input):
        return input

    @staticmethod
    def backward(self, grad_output):
        return grad_output


print(CapsuleConv2d()(input1, input2))
