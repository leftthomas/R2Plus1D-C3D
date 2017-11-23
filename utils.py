import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.svhn import SVHN


def get_iterator(mode, data_type):
    if data_type == 'MNIST':
        data = MNIST(root='data/MNIST', train=mode, transform=transforms.ToTensor(), download=True)
    elif data_type == 'CIFAR10':
        data = CIFAR10(root='data/CIFAR10', train=mode, transform=transforms.ToTensor(), download=True)
    elif data_type == 'CIFAR100':
        data = CIFAR100(root='data/CIFAR100', train=mode, transform=transforms.ToTensor(), download=True)
    elif data_type == 'STL10':
        if mode:
            data = STL10(root='data/STL10', split='train', transform=transforms.ToTensor(), download=True)
        else:
            data = STL10(root='data/STL10', split='test', transform=transforms.ToTensor(), download=True)
    else:
        # SVHN
        if mode:
            data = SVHN(root='data/SVHN', split='train', transform=transforms.ToTensor(), download=True)
        else:
            data = SVHN(root='data/SVHN', split='test', transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=data, batch_size=64, shuffle=mode, num_workers=4)


if __name__ == "__main__":
    t = get_iterator(True, 'MNIST')
    print(t)
