import os
# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, utils

warnings.filterwarnings("ignore")

class MiniImageNet(Dataset):
    """Mini Image Net dataset."""

    def __init__(self, csv_file, separator, root_dir, n_supports, n_queries, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_mappings = pd.read_csv(csv_file, sep=separator, header=None, squeeze=True)
        self.root_dir = root_dir
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_mappings)

    def __getitem__(self, idx):
        drafted_images = []

        #* Randomly pick a class for the support set
        class_name = self.csv_mappings.ix[idx]
        class_path = os.path.join(self.root_dir, class_name)

        #* Get the number of images in the picked class
        images = next(os.walk(class_path))[2]
        n_images = len(images)

        #* Shuffle indices and draw the wanted number of supports
        indices = np.arange(0, n_images)
        np.random.shuffle(indices)
        indices = indices[:(self.n_supports + self.n_queries)]

        for idx in indices:
            img = pil_loader(os.path.join(class_path, images[idx]))
            if self.transforms != None:
                drafted_images.append(self.transforms(img))
            else:
                drafted_images.append(img)

        sample = []
        sample.extend(drafted_images[:self.n_supports])
        sample.extend(drafted_images[self.n_supports:])
        
        return torch.stack(sample), torch.from_numpy(indices)
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
