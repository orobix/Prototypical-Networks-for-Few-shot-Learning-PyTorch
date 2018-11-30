import numpy as np
import torch
from torch.autograd import Variable

from tqdm import tqdm

def meta_test(model, params, use_gpu):
    all_avg_acc = []
    for epoch in range(params.n_epochs):
        progress_bar = tqdm(params.test_loader, desc="Epoch {}".format(epoch))
        progress_bar.set_description("Epoch {}".format(epoch))
        model.eval()

        avg_acc = 0
        for idx, test_batch in enumerate(progress_bar):
            
            inputs, targets = test_batch

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)
            predictions = model(inputs)

            _, test_accuracy = params.criterion(predictions, targets)

            avg_acc += test_accuracy.cpu().data.numpy() / params.n_episodes

        all_avg_acc.append(avg_acc)
    return np.mean(all_avg_acc)
