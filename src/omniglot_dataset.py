from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import cv2
# coding=utf-8
import numpy as np

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
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, mode='train', root='../dataset', transform=None, target_transform=None, download=False):
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        self.all_items = find_classes(
            os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

        self.y = [self.get_path_label(pl)[1] for pl in range(self.__len__())]

        self.x = map(cv2.imread, [self.get_path_label(pl)[0] for pl in range(
            self.__len__())], [cv2.IMREAD_GRAYSCALE] * self.__len__())
        self.x = list(self.x)

        split_size = DSET_SPLIT_SIZES[mode]
        l1 = int(len(self.x) * split_size[0])
        l2 = int(len(self.x) * split_size[1])
        self.x = self.x[l1:l2]
        self.y = self.y[l1:l2]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = cv2.resize(x, (128, 128))
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
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
            os.path.exists(os.path.join(
                self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

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


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx
