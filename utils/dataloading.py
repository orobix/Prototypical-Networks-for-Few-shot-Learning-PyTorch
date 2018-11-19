
import torch

from torchvision import transforms
from utils.mini_image_net import MiniImageNet


def load_split_datasets(paths, n_supports, n_queries, separator=';'):

    trans = transforms.Compose([transforms.ToTensor()])

    train_set = MiniImageNet(csv_file=paths[1],
                            separator=separator,
                            root_dir=paths[0],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=trans)

    valid_set = MiniImageNet(csv_file=paths[2],
                            separator=separator,
                            root_dir=paths[0],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=trans)

    test_set = MiniImageNet(csv_file=paths[3],
                            separator=separator,
                            root_dir=paths[0],
                            n_supports=n_supports,
                            n_queries=n_queries,
                            transforms=trans)

    return train_set, valid_set, test_set

def load_dataloaders(sets, n_ways):
    train_loader = torch.utils.data.DataLoader(sets[0],
                                               batch_size=n_ways,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=False)

    valid_loader = torch.utils.data.DataLoader(sets[1],
                                               batch_size=n_ways,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=False)

    test_loader = torch.utils.data.DataLoader(sets[2],
                                              batch_size=n_ways,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=False)
    return train_loader, valid_loader, test_loader
