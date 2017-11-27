import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.svhn import SVHN


def get_iterator(mode, data_type, using_data_augmentation):
    if using_data_augmentation:
        if data_type == 'MNIST':
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        elif data_type == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
        elif data_type == 'CIFAR100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
        elif data_type == 'STL10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
            ])
        else:
            # SVHN
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    if data_type == 'MNIST':
        if mode:
            data = MNIST(root='data/MNIST', train=mode, transform=transform_train, download=True)
        else:
            data = MNIST(root='data/MNIST', train=mode, transform=transform_test, download=True)

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
            data = STL10(root='data/STL10', split='train', transform=transform_train, download=True)
        else:
            data = STL10(root='data/STL10', split='test', transform=transform_test, download=True)
    else:
        # SVHN
        if mode:
            data = SVHN(root='data/SVHN', split='train', transform=transform_train, download=True)
        else:
            data = SVHN(root='data/SVHN', split='test', transform=transform_test, download=True)
    return DataLoader(dataset=data, batch_size=64, shuffle=mode, num_workers=4)


def get_mean_std(data_type):
    if data_type == 'MNIST':
        train_set = MNIST(root='data/MNIST', train=True, download=True, transform=transforms.ToTensor())
        print(list(train_set.train_data.size()))
        print(train_set.train_data.float().mean() / 255)
        print(train_set.train_data.float().std() / 255)
        # [60000, 28, 28]
        # 0.1306604762738429
        # 0.30810780717887876
    elif data_type == 'CIFAR10':
        train_set = CIFAR10(root='data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
        print(train_set.train_data.shape)
        print(train_set.train_data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.train_data.std(axis=(0, 1, 2)) / 255)
        # (50000, 32, 32, 3)
        # [0.49139968  0.48215841  0.44653091]
        # [0.24703223  0.24348513  0.26158784]
    elif data_type == 'CIFAR100':
        train_set = CIFAR100(root='data/CIFAR100', train=True, download=True, transform=transforms.ToTensor())
        print(train_set.train_data.shape)
        print(train_set.train_data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.train_data.std(axis=(0, 1, 2)) / 255)
        # (50000, 32, 32, 3)
        # [0.50707516  0.48654887  0.44091784]
        # [0.26733429  0.25643846  0.27615047]
    elif data_type == 'STL10':
        train_set = STL10(root='data/STL10', split='train', download=True, transform=transforms.ToTensor())
        print(train_set.data.shape)
        train_set.data = train_set.data.reshape((5000, 3, 96, 96))
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
        # (5000, 3, 96, 96)
        # [0.44671062  0.43980984  0.40664645]
        # [0.26034098  0.25657727  0.27126738]
    else:
        # SVHN
        train_set = SVHN(root='data/SVHN', split='train', download=True, transform=transforms.ToTensor())
        print(train_set.data.shape)
        train_set.data = train_set.data.reshape((73257, 3, 32, 32))
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
        # (73257, 3, 32, 32)
        # [0.4376821   0.4437697   0.47280442]
        # [0.19803012  0.20101562  0.19703614]


if __name__ == "__main__":
    get_mean_std('MNIST')
    get_mean_std('CIFAR10')
    get_mean_std('CIFAR100')
    get_mean_std('STL10')
    get_mean_std('SVHN')
    t = get_iterator(True, 'MNIST', True)
    print(t)
