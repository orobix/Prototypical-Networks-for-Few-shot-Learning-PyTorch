
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from torch.autograd import Variable


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