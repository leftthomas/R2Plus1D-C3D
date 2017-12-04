from torch.autograd import Function


class Conv2dCapsule(Function):
    def __init__(self, weight, in_channels, kernel_size, in_length, out_length, stride,
                 padding, num_iterations):
        super(Conv2dCapsule, self).__init__()
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
        self.input_size = input.size()[-2:]
        return im2col_batch(input, self.kernel_size, self.stride, self.padding)

    @staticmethod
    def backward(self, grad_output):
        return col2im_batch(grad_output, self.kernel_size, self.stride, self.padding, self.input_size)


print(Conv2dCapsule()(input1, input2))
