import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class VideoData(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return self.T(sample), target

    def __len__(self):
        return len(self.samples)


def load_data(batch_size=64):
    data = np.load('data/data_har.npz')
    x_train = data['X_train'].astype(np.float32).reshape((-1, 1, 128, 9))
    y_train = onehot_to_label(data['Y_train'].astype(np.long))
    x_test = data['X_test'].astype(np.float32).reshape((-1, 1, 128, 9))
    y_test = onehot_to_label(data['Y_test'].astype(np.long))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set = VideoData(x_train, y_train, transform)
    test_set = VideoData(x_test, y_test, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        return loss.sum(dim=-1).mean()
