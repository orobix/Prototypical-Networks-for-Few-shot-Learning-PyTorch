# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(model_output, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''

    target_cpu = target.to('cpu')
    model_output_cpu = model_output.to('cpu')

    def get_supports_indexes(c):
        # takes first 5 indexes of class c to make theme supports
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    def get_queries_indexes(c):
        return target_cpu.eq(c).nonzero()[n_support:]

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    support_indexes = list(map(get_supports_indexes, classes))
    supports = [model_output_cpu[idx_list] for idx_list in support_indexes]
    prototypes = torch.stack([support.mean(0) for support in supports]) # mean of supports for each class

    queries_indexes = list(map(get_queries_indexes, classes))
    n_query_per_class = len(queries_indexes[0])

    queries_indexes = torch.stack(queries_indexes).view(-1)
    queries = model_output_cpu[queries_indexes] # here, queries are in groups of 5 (by class)

    dists = euclidean_dist(queries, prototypes)
    log_softmax_dists = F.log_softmax(-dists, dim=1).view(n_classes, n_query_per_class, -1)

    # mask that allows to keep only the log_softmax of distances between query and support of same class
    dist_mask = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query_per_class, 1).long()

    loss_val = -log_softmax_dists.gather(2, dist_mask).squeeze().view(-1).mean()
    _, y_hat = log_softmax_dists.max(2)
    acc_val = y_hat.eq(dist_mask.squeeze()).float().mean()

    return loss_val,  acc_val
