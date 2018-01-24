import argparse

import cv2
import numpy as np
import torch
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from model import models


def processor(sample):
    data, labels, training = sample

    data = Variable(data)
    labels = Variable(labels)
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].data, state['sample'][1])
    confusion_meter.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])

    reset_meters()

    engine.test(processor, utils.get_iterator(False, DATA_TYPE, BATCH_SIZE, USE_DATA_AUGMENTATION))

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())

    print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), 'epochs/epoch_%s_%d.pt' % (DATA_TYPE, state['epoch']))

    # learning rate scheduler
    scheduler.step(meter_loss.value()[0])

    # GradCam visualization
    model.eval()
    original_image, _ = next(iter(utils.get_iterator(False, DATA_TYPE, 16, USE_DATA_AUGMENTATION)))
    data = Variable(original_image)
    if torch.cuda.is_available():
        data = data.cuda()

    cams = []
    for i in range(data.size(0)):
        mask = (grad_cam(data[i].unsqueeze(0))).transpose((1, 2, 0))
        heat_map = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heat_map = np.float32(heat_map) / 255
        img = data[i] - data[i].min()
        img = img / img.max()
        img = img.data.cpu().numpy()
        cam = heat_map + np.float32(img.transpose((1, 2, 0)))
        cam = cam / np.max(cam)
        cams.append(transforms.ToTensor()(np.uint8(255 * cam)))
    cams = torch.stack(cams)
    original_image_logger.log(make_grid(original_image, nrow=4, normalize=True).numpy())
    grad_cam_logger.log(make_grid(cams, nrow=4).numpy())
    model.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Capsule Classfication')
    parser.add_argument('--data_type', default='MNIST', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'STL10'],
                        help='dataset type')
    parser.add_argument('--use_data_augmentation', default='yes', type=str, choices=['yes', 'no'],
                        help='use data augmentation or not')
    parser.add_argument('--batch_size', default=16, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--target_category', default=None, type=int, help='the category of visualization')
    parser.add_argument('--target_layer', default=None, type=int, help='the layer of visualization')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    USE_DATA_AUGMENTATION = True if opt.use_data_augmentation == 'yes' else False
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    TARGET_CATEGORY = opt.target_category
    TARGET_LAYER = opt.target_layer

    class_name = utils.CLASS_NAME[DATA_TYPE]
    CLASSES = 10
    if DATA_TYPE == 'CIFAR100':
        CLASSES = 100

    model = models[DATA_TYPE]()
    loss_criterion = utils.FocalLoss(gamma=2)
    grad_cam = utils.GradCam(model, TARGET_LAYER, TARGET_CATEGORY)
    if torch.cuda.is_available():
        model.cuda()
        loss_criterion.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, threshold=1e-5, verbose=True)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE,
                                    opts={'title': 'Confusion Matrix', 'columnnames': class_name,
                                          'rownames': class_name})
    original_image_logger = VisdomLogger('image', env=DATA_TYPE, opts={'title': 'Original Image'})
    grad_cam_logger = VisdomLogger('image', env=DATA_TYPE, opts={'title': 'GradCam'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, utils.get_iterator(True, DATA_TYPE, BATCH_SIZE, USE_DATA_AUGMENTATION), maxepoch=NUM_EPOCHS,
                 optimizer=optimizer)
