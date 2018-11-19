
import math
import copy
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler

from tqdm import tqdm
from utils.history import History

def train(model, device, optimizer, train_set, valid_set, n_epoch, batch_size, filename, use_gpu=True, scheduler=None, criterion=None):
    history = History()

    train_loader, valid_loader = train_valid_loaders(train_set, valid_set, batch_size)

    best_acc = 0.0
    for epoch in range(n_epoch):
        progress_bar = tqdm(train_loader, desc="Epoch {}".format(epoch))

        do_epoch(device, criterion, model, optimizer, scheduler, use_gpu, epoch, batch_size, round(len(train_set)/batch_size), progress_bar)

        train_acc, train_loss = validate(model, criterion, train_loader, use_gpu)
        val_acc, val_loss = validate(model, criterion, valid_loader, use_gpu)

        if scheduler:
#            scheduler.step(val_acc)
             scheduler.step(val_loss)
        print('Accuracy: %2f' % val_acc)

        # Deep copy the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, filename)

        history.save(train_acc, val_acc, train_loss, val_loss, optimizer.param_groups[0]['lr'])

    return history


def do_epoch(device, criterion, model, optimizer, scheduler, use_gpu, epoch, batch_size, set_size, progress_bar):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description("Epoch {}".format(epoch))

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)
        output = model(inputs)

        optimizer.zero_grad()
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.cpu().data.numpy()})


def validate(model, criterion, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    model.eval()

    for idx, (inputs, targets) in enumerate(val_loader):

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        output = model(inputs)

        predictions = output.max(dim=1)[1]
        loss = criterion(output, targets).data[0]

        val_loss.append(loss)
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def test(model, criterion, test_dataset, batch_size, use_gpu=True):
    sampler = RandomSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler)

    score, loss = validate(model, criterion, test_loader, use_gpu=use_gpu)
    return score

def train_valid_loaders(train_set, valid_set, batch_size):
    train_sampler = RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=8,
                                                pin_memory=True)

    valid_sampler = RandomSampler(valid_set)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=8,
                                               pin_memory=True)
    return train_loader, valid_loader

