import numpy as np


class PrototypicalBatchSampler(object):

    def __init__(self, labels, classes_per_it, num_support, num_query, iterations):
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_support + num_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.idxs = range(len(self.labels))
        self.ndclasses = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        # TODO avoid for (?)
        for idx, label in enumerate(self.labels):
            self.ndclasses[label, np.where(np.isnan(self.ndclasses[label]))[0][0]] = idx

    def __iter__(self):
        for it in range(self.iterations):
            batch = np.zeros((self.sample_per_class * self.classes_per_it), dtype=int)
            curr_classes = np.random.choice(self.classes, self.classes_per_it, replace=False)
            for i, c in enumerate(curr_classes):
                s = slice(i*self.sample_per_class, (i+1)*self.sample_per_class)
                batch[s] = np.random.choice(self.ndclasses[c], self.sample_per_class, replace=False)
            np.random.shuffle(batch) # this should be uselsess (?)
            yield batch

    def __len__(self):
        return self.iterations
