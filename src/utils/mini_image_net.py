import os
import warnings # Ignore warnings

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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
        self.all_targets = []
        self.all_image_tensors = []

        n_class = len(all_class_paths)

        for i, current_class_dir in enumerate(all_class_paths):
            print("debut du load des images de la classe {} sur {} en RAM".format(i, n_class))

            full_file_paths_of_class = next(os.walk(current_class_dir))[2]
            full_file_paths_of_class = [os.path.join(current_class_dir, image_filename)
                                       for image_filename
                                       in full_file_paths_of_class]

            target_of_class = [i] * len(full_file_paths_of_class)

            self.all_targets.extend(target_of_class)

            image_tensors_from_path = [pil_loader(full_path) for full_path in full_file_paths_of_class]
            image_tensors_from_path = [self.transforms(image_obj) for image_obj in image_tensors_from_path]

            self.all_image_tensors.extend(image_tensors_from_path)

        self.all_targets = torch.LongTensor(self.all_targets)
        self.all_image_tensors = torch.stack(self.all_image_tensors)

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, idx):

        x = self.all_image_tensors[idx]
        y = self.all_targets[idx]
        
        return x, y


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
