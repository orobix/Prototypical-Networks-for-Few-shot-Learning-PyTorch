import torch

from utils.mini_image_net import MiniImageNet
from utils.prototypical_batch_sampler import PrototypicalBatchSampler


def load_meta_train_set(paths, separator=';'):
    train_set = MiniImageNet(csv_file=paths['train_dir'],
                             separator=separator,
                             root_dir=paths['root_dir'])

    valid_set = MiniImageNet(csv_file=paths['valid_dir'],
                             separator=separator,
                             root_dir=paths['root_dir'])

    return train_set, valid_set


def load_meta_test_set(paths, separator=';'):
    test_set = MiniImageNet(csv_file=paths['test_dir'],
                            separator=separator,
                            root_dir=paths['root_dir'])
    return test_set


def load_meta_train_dataloaders(sets, samples_per_class, n_episodes, classes_per_it):
    train_set = sets['train_set']
    valid_set = sets['valid_set']

    train_sampler = PrototypicalBatchSampler(labels=train_set.all_targets,
                                             classes_per_it=classes_per_it[0],
                                             num_samples=samples_per_class,
                                             iterations=n_episodes)

    valid_sampler = PrototypicalBatchSampler(labels=valid_set.all_targets,
                                             classes_per_it=classes_per_it[1],
                                             num_samples=samples_per_class,
                                             iterations=n_episodes)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_sampler=valid_sampler)

    return train_loader, valid_loader


def load_meta_test_dataloader(test_set, samples_per_class, n_episodes, classes_per_it):
    test_sampler = PrototypicalBatchSampler(labels=test_set.all_targets,
                                            classes_per_it=classes_per_it,
                                            num_samples=samples_per_class,
                                            iterations=n_episodes)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_sampler=test_sampler)

    return test_loader


def get_training_and_validation_sets(paths):
    train_set, valid_set = load_meta_train_set(paths)
    sets = {'train_set': train_set,
            'valid_set': valid_set}
    return sets
