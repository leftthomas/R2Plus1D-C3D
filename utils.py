import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.stl10 import STL10

import config


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


def get_iterator(mode):
    tr = transforms.Compose([transforms.Scale(size=24), transforms.ToTensor()])
    # data = CIFAR10(root='data/CIFAR10', train=mode, transform=tr, download=True)
    # data = CIFAR100(root='data/CIFAR100', train=mode, transform=tr, download=True)
    if mode:
        data = STL10(root='data/STL10', split='train', transform=tr, download=True)
    else:
        data = STL10(root='data/STL10', split='test', transform=tr, download=True)
    return DataLoader(dataset=data, batch_size=config.BATCH_SIZE, shuffle=mode, num_workers=4)


if __name__ == "__main__":
    t = get_iterator(True)
    print(t)
