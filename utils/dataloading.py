
import torch

from utils.mini_image_net import MiniImageNet


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

def load_dataloaders(sets, n_ways):
    train_loader = torch.utils.data.DataLoader(sets['train_set'],
                                               batch_size=n_ways,
                                               shuffle=True)
                                               #num_workers=1,
                                               #pin_memory=False)

    valid_loader = torch.utils.data.DataLoader(sets['valid_set'],
                                               batch_size=n_ways,
                                               shuffle=True)
                                               #num_workers=1,
                                               #pin_memory=False)

    test_loader = torch.utils.data.DataLoader(sets['test_set'],
                                              batch_size=n_ways,
                                              shuffle=True)
                                              #num_workers=1,
                                              #pin_memory=False)
    return train_loader, valid_loader, test_loader
