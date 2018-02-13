# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import errno
import os
import re
import cv2

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

DSET_SPLIT_SIZES = {
    'train': [0, 1],
    'val': [0, 1],
    'test': [0, 0],
    'pred': [0, 1],
    None: [0, 1],
}


class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'processed'

    def __init__(self, mode='train', root='../dataset', transform=None, target_transform=None, download=True):

        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        self.classes = get_current_classes(os.path.join(self.root, self.splits_folder, mode + '.txt'))

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder), self.classes)
        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl) for pl in range(len(self))])

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = np.expand_dims(x, 2)
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        images_evaluation_path = os.path.join(self.root, self.processed_folder, "images_evaluation")
        images_background_path = os.path.join(self.root, self.processed_folder, "images_background")
        return os.path.exists(images_evaluation_path) and \
            os.path.exists(images_background_path)

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir, classes):
    retour = []
    rots = [0, 90, 180, 270]
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split('/')
            lr = len(r)
            label = r[lr - 2] + "/" + r[lr - 1]
            if label in classes and (f.endswith("png")):
                retour.extend([(f, label, root, rot) for rot in rots])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    f = open(fname)
    classes = list(set([re.match('.*/', line).group(0)[:-1] for line in f]))
    f.close()
    return classes


def load_img(path, idx):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = np.rot90(x, k=(idx % 4 * 90))
    x = cv2.resize(x, (28, 28))
    return x
