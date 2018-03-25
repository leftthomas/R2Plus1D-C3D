from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet20


class CIFAR100CapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(CIFAR100CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for name, module in resnet20().named_children():
            if name == 'conv1' or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=256, out_capsules=100, in_length=4, out_length=8,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.pool(out)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 4)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = CIFAR100CapsuleNet()
    for m in model.named_children():
        print(m)
