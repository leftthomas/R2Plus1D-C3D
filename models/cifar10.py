from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet110


class CIFAR10CapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(CIFAR10CapsuleNet, self).__init__()

        layers = []
        for name, module in resnet110().named_children():
            if isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=16, out_capsules=10, in_length=4, out_length=16,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 4)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = CIFAR10CapsuleNet()
    for m in model.named_children():
        print(m)
