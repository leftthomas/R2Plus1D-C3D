import argparse

import pandas as pd
import torch
import torchnet as tnt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from model import Model
from utils import MarginLoss, AudioDataset


def processor(sample):
    data, label, training = sample
    labels = torch.eye(10).index_select(dim=0, index=label)

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
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][1])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])

    print('[Epoch %d] Training Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    # save model at each epoch
    torch.save(model.state_dict(), 'epochs/%d.pth' % state['epoch'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Audio Classification')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--batch_size', default=20, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')

    opt = parser.parse_args()

    NUM_ITERATIONS, BATCH_SIZE, NUM_EPOCHS = opt.num_iterations, opt.batch_size, opt.num_epochs

    # load data
    print('loading data...')
    train_dataset, test_dataset = AudioDataset(data_type='train'), AudioDataset(data_type='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model, loss_criterion = Model(NUM_ITERATIONS), MarginLoss()
    if torch.cuda.is_available():
        model, loss_criterion = model.to('cuda'), loss_criterion.to('cuda')

    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    results = {'train_loss': [], 'train_accuracy': []}

    optimizer = Adam(model.parameters())

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

    # save statistics at end
    data_frame = pd.DataFrame(data={'train_loss': results['train_loss'], 'train_accuracy': results['train_accuracy']},
                              index=range(1, NUM_EPOCHS + 1))
    data_frame.to_csv('statistics/results.csv', index_label='epoch')

    # test
    with torch.no_grad():
        for data in test_loader:
            if torch.cuda.is_available():
                data = data.to('cuda')
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
