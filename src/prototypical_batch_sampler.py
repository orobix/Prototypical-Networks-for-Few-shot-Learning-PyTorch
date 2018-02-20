# coding=utf-8
import numpy as np


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it', 'num_support', 'num_query',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'iterations').
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
        self.idxs = range(len(self.labels))
        self.ndclasses = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        for idx, label in enumerate(self.labels):
            self.ndclasses[label, np.where(np.isnan(self.ndclasses[label]))[0][0]] = idx

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        for it in range(self.iterations):
            batch = np.zeros(
                (self.sample_per_class * self.classes_per_it), dtype=int)
            curr_classes = np.random.choice(
                self.classes, self.classes_per_it, replace=False)
            for i, c in enumerate(curr_classes):
                s = slice(i * self.sample_per_class,
                          (i + 1) * self.sample_per_class)
                batch[s] = np.random.choice(
                    self.ndclasses[c], self.sample_per_class, replace=False)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
