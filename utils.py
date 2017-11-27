import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.svhn import SVHN


def get_iterator(mode, data_type):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_type == 'MNIST':
        data = MNIST(root='data/MNIST', train=mode, transform=transforms.ToTensor(), download=True)

    elif data_type == 'CIFAR10':
        if mode:
            data = CIFAR10(root='data/CIFAR10', train=mode, transform=transform_train, download=True)
        else:
            data = CIFAR10(root='data/CIFAR10', train=mode, transform=transform_test, download=True)

    elif data_type == 'CIFAR100':
        if mode:
            data = CIFAR100(root='data/CIFAR100', train=mode, transform=transform_train, download=True)
        else:
            data = CIFAR100(root='data/CIFAR100', train=mode, transform=transform_test, download=True)

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
