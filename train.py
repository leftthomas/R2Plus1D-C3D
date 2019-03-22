import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchnet as tnt
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from model import Model
from utils import Indegree, MarginLoss


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


def on_forward(state):
    meter_loss.add(state['loss'].detach().cpu().item())
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][0].y)


def on_start_epoch(state):
    reset_meters()


def on_end_epoch(state):
    train_loss_logger.log(state['epoch'], meter_loss.value()[0], name='fold_' + str(fold_number))
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='fold_' + str(fold_number))
    fold_results['train_loss'].append(meter_loss.value()[0])
    fold_results['train_accuracy'].append(meter_accuracy.value()[0])

    # val
    reset_meters()
    with torch.no_grad():
        engine.test(processor, val_loader)

    val_loss_logger.log(state['epoch'], meter_loss.value()[0], name='fold_' + str(fold_number))
    val_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='fold_' + str(fold_number))
    fold_results['val_loss'].append(meter_loss.value()[0])
    fold_results['val_accuracy'].append(meter_accuracy.value()[0])

    # test
    reset_meters()
    with torch.no_grad():
        engine.test(processor, test_loader)

    test_loss_logger.log(state['epoch'], meter_loss.value()[0], name='fold_' + str(fold_number))
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='fold_' + str(fold_number))
    fold_results['test_loss'].append(meter_loss.value()[0])
    fold_results['test_accuracy'].append(meter_accuracy.value()[0])

    # save model at each fold
    torch.save(model.state_dict(), 'epochs/%s_%d.pth' % (DATA_TYPE, fold_number))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_type', default='MUTAG', type=str,
                        choices=['MUTAG', 'PTC_MR', 'NCI1', 'NCI109', 'PROTEINS', 'DD', 'ENZYMES', 'COLLAB',
                                 'IMDB-BINARY', 'IMDB-MULTI'], help='dataset type')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=20, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    NUM_ITERATIONS = opt.num_iterations
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    data_set = TUDataset('data/%s' % DATA_TYPE, DATA_TYPE, pre_transform=Indegree(), use_node_attr=True)
    NUM_FEATURES, NUM_CLASSES = data_set.num_features, data_set.num_classes
    print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d' % (data_set, NUM_FEATURES, NUM_CLASSES))

    over_results = {'train_accuracy': [], 'val_accuracy': [], 'test_accuracy': []}

    model = Model(NUM_FEATURES, NUM_CLASSES, NUM_ITERATIONS)
    loss_criterion = MarginLoss()
    if torch.cuda.is_available():
        model = model.to('cuda')
        loss_criterion = loss_criterion.to('cuda')

    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    train_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Train Accuracy'})
    val_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Val Loss'})
    val_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Val Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Test Accuracy'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    # create a 10-fold cross validation
    train_iter = tqdm(range(1, 11), desc='Training Model......')
    for fold_number in train_iter:
        # 90/10 train/test split
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (DATA_TYPE, fold_number),
                                 dtype=np.int32).tolist()
        # randomly sample 10% from train split as val split
        val_idxes = random.sample(train_idxes, int(len(train_idxes) * 0.1))
        train_idxes = list(set(train_idxes) - set(val_idxes))
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (DATA_TYPE, fold_number),
                                dtype=np.int32).tolist()
        train_idxes = torch.as_tensor(train_idxes, dtype=torch.long)
        val_idxes = torch.as_tensor(val_idxes, dtype=torch.long)
        test_idxes = torch.as_tensor(test_idxes, dtype=torch.long)

        train_set, val_set, test_set = data_set[train_idxes], data_set[val_idxes], data_set[test_idxes]
        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_idxes, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

        fold_results = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                        'test_accuracy': []}

        optimizer = Adam(model.parameters())

        engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
        # save statistics at every fold
        fold_data_frame = pd.DataFrame(
            data={'train_loss': fold_results['train_loss'], 'val_loss': fold_results['val_loss'],
                  'test_loss': fold_results['test_loss'], 'train_accuracy': fold_results['train_accuracy'],
                  'val_accuracy': fold_results['val_accuracy'], 'test_accuracy': fold_results['test_accuracy']},
            index=range(1, NUM_EPOCHS + 1))
        fold_data_frame.to_csv('statistics/%s_results_%d.csv' % (DATA_TYPE, fold_number), index_label='epoch')

        # record the results when the model obtains best result on val split
        best_index = fold_results['val_accuracy'].index(max(fold_results['val_accuracy']))
        over_results['train_accuracy'].append(fold_results['train_accuracy'][best_index])
        over_results['val_accuracy'].append(fold_results['val_accuracy'][best_index])
        over_results['test_accuracy'].append(fold_results['test_accuracy'][best_index])

        train_iter.set_description(
            '[Fold %d] Training Accuracy: %.2f%% Valing Accuracy: %.2f%% Testing Accuracy: %.2f%%' % (
                fold_number, fold_results['train_accuracy'][best_index], fold_results['val_accuracy'][best_index],
                fold_results['test_accuracy'][best_index]))

        # reset model for each fold
        model = Model(NUM_FEATURES, NUM_CLASSES, NUM_ITERATIONS)
        if torch.cuda.is_available():
            model = model.to('cuda')

    # save statistics at all fold
    data_frame = pd.DataFrame(
        data={'train_accuracy': over_results['train_accuracy'], 'val_accuracy': over_results['val_accuracy'],
              'test_accuracy': over_results['test_accuracy']}, index=range(1, 11))
    data_frame.to_csv('statistics/%s_results_overall.csv' % DATA_TYPE, index_label='fold')

    print('Overall Training Accuracy: %.2f%% (std: %.2f) Valing Accuracy: %.2f%% (std: %.2f) '
          'Testing Accuracy: %.2f%% (std: %.2f)' % (np.array(over_results['train_accuracy']).mean(),
                                                    np.array(over_results['train_accuracy']).std(),
                                                    np.array(over_results['val_accuracy']).mean(),
                                                    np.array(over_results['val_accuracy']).std(),
                                                    np.array(over_results['test_accuracy']).mean(),
                                                    np.array(over_results['test_accuracy']).std()))
