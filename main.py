#%%
import copy
import os
import pdb
import time

import torch
from torch import load, nn, optim

from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
from utils.transforms import ToTensor
from utils.miniimagenet_dataset import MiniImageNet

#################################
#           Variables           #
#################################
# Constants
use_gpu = torch.cuda.is_available()
dataset_path = './mini_imagenet/images'
train_path =  './mini_imagenet/csvsplits/train.csv'
valid_path = './mini_imagenet/csvsplits/valid.csv'
test_path = './mini_imagenet/csvsplits/test.csv'
separator = ';'

############################# 
#       Hyperparameters     #
#############################
n_ways = 5
n_shots = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################
#     Chargement du dataset     #
#################################

transforms = transforms.Compose([
                      # transforms.Resize((84,84), interpolation=2),
                      transforms.ToTensor()
             ])

train_set = MiniImageNet(csv_file=train_path,
                        separator=separator,
                        root_dir=dataset_path,
                        n_shots=n_shots,
                        transforms=transforms)

valid_set = MiniImageNet(csv_file=valid_path,
                        separator=separator,
                        root_dir=dataset_path,
                        n_shots=n_shots,
                        transforms=transforms)

test_set = MiniImageNet(csv_file=test_path,
                        separator=separator,
                        root_dir=dataset_path,
                        n_shots=n_shots,
                        transforms=transforms)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=n_ways,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=False)

for idx, (query, support) in enumerate(train_loader):
  print('Idx: ' + idx)
  print('Query: ' + query)
  print('Support :' + support)
  print('stop')

# if torch.cuda.is_available():
    # net = net.cuda()

# print("\n\n Quantity of parameters: ", sum([param.nelement() for param in net.parameters()]))


#################################
#   Parametres d'entrainement   #
#################################
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
#                           lr=learning_rate,
#                           momentum=momentum,
#                           nesterov=True,
#                           weight_decay=0.01)

# criterion = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                  mode='min',
#                                                  patience=patience,
#                                                  verbose=True,
#                                                  threshold=accuracy_threshold)

# #################################
# #          Entrainement         #
# #################################
# history = train(net, device, optimizer, train_set, valid_set, n_epoch, batch_size, save_path, use_gpu=use_gpu, criterion=criterion, scheduler=scheduler)


# #################################
# #             Tests             #
# #################################
# # Charger le meilleur modele enregistre
# state_dict = load(save_path)
# net.load_state_dict(state_dict)
# print('Précision en test: {:.2f}'.format(test(net, criterion, test_set, batch_size)))
