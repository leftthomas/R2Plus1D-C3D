from capsule_layer import CapsuleLinear
from torch import nn
from torchvision.models.resnet import BasicBlock


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, planes=64, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, planes=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes=512, blocks=2, stride=2)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=128, out_capsules=32, in_length=4, out_length=8,
                                                      routing_type='contract', share_weight=True,
                                                      num_iterations=num_iterations),
                                        CapsuleLinear(in_capsules=32, out_capsules=10, in_length=8, out_length=16,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = x.view(*x.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 4)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
