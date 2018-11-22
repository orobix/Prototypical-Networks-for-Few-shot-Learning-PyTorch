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

        all_class_paths = [os.path.join(self.root_dir, class_name) for class_name in self.csv_mappings]
        self.all_image_names = []
        self.all_targets = []
        self.all_images_tensors = []

        n_class = len(all_class_paths)

        for i, path in enumerate(all_class_paths):
            print("debut du load des images de la classe {} sur {} en RAM".format(i, n_class))
            images_from_path = next(os.walk(path))[2]
            images_from_path = [os.path.join(path, image_name) for image_name in images_from_path]
            target_of_path = [i] * len(images_from_path)
            self.all_image_names.extend(images_from_path)
            self.all_targets.extend(target_of_path)

            image_tensors_from_path = [pil_loader(full_name) for full_name in images_from_path]
            image_tensors_from_path = [self.transforms(image_obj) for image_obj in image_tensors_from_path]
            self.all_images_tensors.extend(image_tensors_from_path)

        self.all_targets = torch.LongTensor(self.all_targets)
        self.all_images_tensors = torch.stack(self.all_images_tensors)
        

    def __len__(self):
        return len(self.csv_mappings)

    def __getitem__(self, idx):

        x = self.all_images_tensors[idx]
        y = self.all_targets[idx]
        
        return x, y
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
