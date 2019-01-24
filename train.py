import argparse

import pandas as pd
import torch
import torchnet as tnt
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

from model import Model
from utils import MarginLoss


def processor(sample):
    data, training = sample
    labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=data.y)

    if torch.cuda.is_available():
        data, labels = data.to('cuda'), labels.to('cuda')

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'] = state['sample'], state['train']


def reset_meters():
    meter_loss.reset()
    meter_accuracy.reset()
    meter_confusion.reset()


def on_forward(state):
    meter_loss.add(state['loss'].detach().cpu().item())
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][0].y)
    meter_confusion.add(state['output'].detach().cpu(), state['sample'][0].y)


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f Training Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])

    reset_meters()
    with torch.no_grad():
        engine.test(processor, test_loader)

    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(meter_confusion.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_accuracy'].append(meter_accuracy.value()[0])

    print('[Epoch %d] Testing Loss: %.4f Testing Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    # save best model
    global best_accuracy
    if meter_accuracy.value()[0] > best_accuracy:
        torch.save(model.state_dict(), 'epochs/%s.pth' % DATA_TYPE)
        best_accuracy = meter_accuracy.value()[0]

    # save statistics at every 10 epochs
    if state['epoch'] % 10 == 0:
        data_frame = pd.DataFrame(
            data={'train_loss': results['train_loss'], 'test_loss': results['test_loss'],
                  'train_accuracy': results['train_accuracy'],
                  'test_accuracy': results['test_accuracy']},
            index=range(1, state['epoch'] + 1))
        data_frame.to_csv('statistics/%s_results.csv' % DATA_TYPE, index_label='epoch')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='DD', type=str,
                        choices=['REDDIT-BINARY', 'DD', 'REDDIT-MULTI-12K', 'REDDIT-MULTI-5K', 'PTC_MR', 'NCI1',
                                 'NCI109', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'ENZYMES', 'COLLAB'],
                        help='dataset type')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    results = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
    # record current best measures
    best_accuracy = 0

    data_set = TUDataset('data/%s' % DATA_TYPE, DATA_TYPE).shuffle()
    NUM_FEATURES, NUM_CLASSES = data_set.num_features, data_set.num_classes
    # create a 90/10 train/test split
    train_len = int(0.9 * len(data_set))
    train_set, test_set = data_set[:train_len], data_set[train_len:]
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(NUM_FEATURES, NUM_CLASSES, NUM_ITERATIONS)
    loss_criterion = MarginLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')
        loss_criterion = loss_criterion.to('cuda')

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', env=DATA_TYPE, opts={'title': 'Confusion Matrix'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
