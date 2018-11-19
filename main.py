#%%
import copy
import os

import torch
from torch import load, nn, optim
from torch.utils.data.sampler import RandomSampler

from torchvision import datasets, transforms

from protonet import ProtoNet

from utils.dataloading import load_split_datasets, load_dataloaders

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

paths = [dataset_path, train_path, valid_path, test_path]
train_set, valid_set, test_set = load_split_datasets(paths, n_shots)

sets = [train_set, valid_set, test_set]
train_loader, valid_loader, test_loader = load_dataloaders(sets, n_ways)

model = ProtoNet()

if torch.cuda.is_available():
    model = model.cuda()

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
