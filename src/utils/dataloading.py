
import torch

from utils.mini_image_net import MiniImageNet
from utils.prototypical_batch_sampler import PrototypicalBatchSampler


def load_split_datasets(paths, n_supports, n_queries, separator=';', transforms=None):
    train_set = MiniImageNet(csv_file=paths['train_dir'],
                            separator=separator,
                            root_dir=paths['root_dir'],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=transforms)

    valid_set = MiniImageNet(csv_file=paths['valid_dir'],
                            separator=separator,
                            root_dir=paths['root_dir'],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=transforms)

    test_set = MiniImageNet(csv_file=paths['test_dir'],
                            separator=separator,
                            root_dir=paths['root_dir'],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=transforms)

    return train_set, valid_set, test_set


def load_dataloaders(sets, samples_per_class=10, num_it=100, classes_per_it=(60, 10, 10)):

    train_set = sets['train_set']
    valid_set = sets['valid_set']
    test_set = sets['test_set']

    train_sampler = PrototypicalBatchSampler(labels=train_set.all_targets,
                                         classes_per_it=classes_per_it[0],
                                         num_samples=samples_per_class,
                                         iterations=num_it)

    valid_sampler = PrototypicalBatchSampler(labels=valid_set.all_targets,
                                         classes_per_it=classes_per_it[1],
                                         num_samples=samples_per_class,
                                         iterations=num_it)

    test_sampler = PrototypicalBatchSampler(labels=test_set.all_targets,
                                         classes_per_it=classes_per_it[1],
                                         num_samples=samples_per_class,
                                         iterations=num_it)

    train_loader = torch.utils.data.DataLoader(sets['train_set'],
                                               batch_sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(sets['valid_set'],
                                               batch_sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(sets['test_set'],
                                              batch_sampler=test_sampler)

    return train_loader, valid_loader, test_loader
