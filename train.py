import argparse

import pandas as pd
import torch
import torch.optim as optim
import torchnet as tnt
from torch import nn
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

import utils
from model import Model


def processor(sample):
    data, labels, training = sample

    data, labels = data.cuda(), labels.cuda()

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    meter_confusion.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][1])
    meter_confusion.add(state['output'].detach().cpu(), state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    loss_logger.log(state['epoch'], meter_loss.value()[0], name='train')
    top1_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='train')
    top5_accuracy_logger.log(state['epoch'], meter_accuracy.value()[1], name='train')
    train_confusion_logger.log(meter_confusion.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_top1_accuracy'].append(meter_accuracy.value()[0])
    results['train_top5_accuracy'].append(meter_accuracy.value()[1])
    print('[Epoch %d] Training Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1]))

    reset_meters()

    with torch.no_grad():
        engine.test(processor, val_loader)

    loss_logger.log(state['epoch'], meter_loss.value()[0], name='val')
    top1_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='val')
    top5_accuracy_logger.log(state['epoch'], meter_accuracy.value()[1], name='val')
    val_confusion_logger.log(meter_confusion.value())
    results['val_loss'].append(meter_loss.value()[0])
    results['val_top1_accuracy'].append(meter_accuracy.value()[0])
    results['val_top5_accuracy'].append(meter_accuracy.value()[1])
    print('[Epoch %d] Valing Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1]))

    # save best model
    global best_accuracy
    if meter_accuracy.value()[0] > best_accuracy:
        torch.save(model.state_dict(), 'epochs/{}.pth'.format(DATA_TYPE))
        best_accuracy = meter_accuracy.value()[0]

    reset_meters()

    with torch.no_grad():
        engine.test(processor, test_loader)

    loss_logger.log(state['epoch'], meter_loss.value()[0], name='test')
    top1_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='test')
    top5_accuracy_logger.log(state['epoch'], meter_accuracy.value()[1], name='test')
    test_confusion_logger.log(meter_confusion.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_top1_accuracy'].append(meter_accuracy.value()[0])
    results['test_top5_accuracy'].append(meter_accuracy.value()[1])
    print('[Epoch %d] Testing Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1]))

    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(
            data={'train_loss': results['train_loss'], 'train_top1_accuracy': results['train_top1_accuracy'],
                  'train_top5_accuracy': results['train_top5_accuracy'], 'val_loss': results['val_loss'],
                  'val_top1_accuracy': results['val_top1_accuracy'], 'val_top5_accuracy': results['val_top5_accuracy'],
                  'test_loss': results['test_loss'], 'test_top1_accuracy': results['test_top1_accuracy'],
                  'test_top5_accuracy': results['test_top5_accuracy']},
            index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_TYPE), index_label='epoch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Activity Recognition Model')
    parser.add_argument('--data_type', default='ucf101', type=str, choices=['ucf101', 'hmdb51', 'kinetics600'],
                        help='dataset type')
    parser.add_argument('--gpu_ids', default='0,1,2', type=str, help='selected gpu')
    parser.add_argument('--batch_size', default=30, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='training epoch number')

    opt = parser.parse_args()
    DATA_TYPE = opt.data_type
    GPU_IDS = opt.gpu_ids
    BATCH_SIZE = opt.batch_size
    NUM_EPOCH = opt.num_epochs
    device_ids = [int(gpu) for gpu in GPU_IDS.split(',')]
    results = {'train_loss': [], 'train_top1_accuracy': [], 'train_top5_accuracy': [], 'val_loss': [],
               'val_top1_accuracy': [], 'val_top5_accuracy': [], 'test_loss': [], 'test_top1_accuracy': [],
               'test_top5_accuracy': []}
    # record best val accuracy
    best_accuracy = 0

    train_loader, val_loader, test_loader = utils.load_data(DATA_TYPE, BATCH_SIZE)
    NUM_CLASS = len(train_loader.dataset.label2index)
    model = Model(NUM_CLASS)
    if len(device_ids) > 1:
        if torch.cuda.device_count() >= len(device_ids):
            model = nn.DataParallel(model, device_ids=device_ids)
        else:
            raise ValueError("the machine don't have {} gpus".format(str(len(device_ids))))
    else:
        model = model.cuda(device_ids[0])
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    print("Number of parameters:", sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(NUM_CLASS, normalized=True)

    loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Loss'})
    top1_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Top1 Accuracy'})
    top5_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Top5 Accuracy'})
    train_confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE, opts={'title': 'Train Confusion Matrix'})
    val_confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE, opts={'title': 'Val Confusion Matrix'})
    test_confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE, opts={'title': 'Test Confusion Matrix'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCH, optimizer=optimizer)
