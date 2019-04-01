import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

import utils
from model import Network

torch.manual_seed(10)


def processor(sample):
    data, labels, training = sample

    data, labels = data.to(DEVICE), labels.to(DEVICE)

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
    accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='train')
    train_confusion_logger.log(meter_confusion.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])
    print('[Epoch %d] Training Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    reset_meters()

    with torch.no_grad():
        engine.test(processor, test_loader)

    loss_logger.log(state['epoch'], meter_loss.value()[0], name='test')
    accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='test')
    test_confusion_logger.log(meter_confusion.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_accuracy'].append(meter_accuracy.value()[0])
    print('[Epoch %d] Testing Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    # save model
    torch.save(model.state_dict(), 'epochs/%d.pth' % (state['epoch']))
    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(
            data={'train_loss': results['train_loss'], 'train_accuracy': results['train_accuracy'],
                  'test_loss': results['test_loss'], 'test_accuracy': results['test_accuracy']},
            index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/results.csv', index_label='epoch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Activity Recognition Model')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')

    opt = parser.parse_args()

    BATCH_SIZE = opt.batch_size
    NUM_EPOCH = opt.num_epochs
    NUM_CLASS = 6
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    train_loader, test_loader = utils.load_data(batch_size=BATCH_SIZE)
    model = Network().to(DEVICE)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(NUM_CLASS, normalized=True)

    loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})
    accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Accuracy'})
    train_confusion_logger = VisdomLogger('heatmap', opts={'title': 'Train Confusion Matrix'})
    test_confusion_logger = VisdomLogger('heatmap', opts={'title': 'Test Confusion Matrix'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCH, optimizer=optimizer)
