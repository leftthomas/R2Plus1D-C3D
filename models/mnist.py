from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet110


class MNISTCapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(MNISTCapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for name, module in resnet110().named_children():
            if name == 'conv1' or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=32, out_capsules=16, in_length=2, out_length=4,
                                                      routing_type='contract', share_weight=True,
                                                      num_iterations=num_iterations),
                                        CapsuleLinear(in_capsules=16, out_capsules=10, in_length=4, out_length=8,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.pool(out)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 2)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = MNISTCapsuleNet()
    for m in model.named_children():
        print(m)
