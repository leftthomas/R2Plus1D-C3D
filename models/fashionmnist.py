from torch import nn


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum'):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 128, out_features=256),
            nn.Linear(in_features=256, out_features=10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        out = self.classifier(out).unsqueeze(dim=-1)
        classes = out.norm(dim=-1)
        return classes
