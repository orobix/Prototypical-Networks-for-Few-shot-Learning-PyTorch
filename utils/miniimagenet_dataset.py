import os
# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, utils

warnings.filterwarnings("ignore")

class MiniImageNet(Dataset):
    """Mini Image Net dataset."""

    def __init__(self, csv_file, separator, root_dir, n_shots, transforms=None):
      """
      Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
      """
      self.csv_mappings = pd.read_csv(csv_file, sep=separator, header=None, squeeze=True)
      self.root_dir = root_dir
      self.n_shots = n_shots
      self.transforms = transforms

    def __len__(self):
      return len(self.csv_mappings)

    def __getitem__(self, idx):
      shots = []

      #* Start by randomly picking a class
      class_name = self.csv_mappings.ix[idx]
        
      #* Get the folder of the picked class
      class_path = os.path.join(self.root_dir, class_name)

      #* Get the number of images of the picked class
      images = next(os.walk(class_path))[2]
      n_images = len(images)

      #* Shuffle indices and draw the wanted number of Shots
      indices = np.arange(0, n_images)
      np.random.shuffle(indices)
      indices = indices[:self.n_shots * 2]

      drafted_images = []
      for i in indices:
        drafted_images.append(images[i])

      #* Repeat for the wanted number of Ways
      for drafted_image in drafted_images:
        img = Image.open(os.path.join(class_path, drafted_image))
        shots.append(self.transforms(img))

      query_vector = shots[:self.n_shots]
      support_vector = shots[self.n_shots:]
      sample = [query_vector, support_vector]

      return sample
