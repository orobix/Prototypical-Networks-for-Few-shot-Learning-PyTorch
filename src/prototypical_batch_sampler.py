# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it', 'num_support', 'num_query',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_support, num_query, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_support: number of support samples for each iteration for each class
        - num_query: number of query samples for each iteration for each class
        - iterations:number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_support + num_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.label_tens = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.label_tens = torch.Tensor(self.label_tens)
        for idx, label in enumerate(self.labels):
            self.label_tens[label, np.where(np.isnan(self.label_tens[label]))[0][0]] = idx

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        cpi = self.classes_per_it
        spc = self.sample_per_class

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                sample_idxs = torch.randperm(len(self.label_tens[c]))[:spc]
                batch[s] = self.label_tens[c][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
