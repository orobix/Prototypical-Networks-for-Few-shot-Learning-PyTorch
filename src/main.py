#%%
import copy
import os

import torch
from torch import load, nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from protonet import ProtoNet
from prototypical_loss import PrototypicalLoss
from torchvision import datasets, transforms
from utils.dataloading import load_dataloaders, load_split_datasets
from utils.graphs import History

#*################################
#*           Variables           #
#*################################
# Constants
use_gpu = torch.cuda.is_available()
dataset_path = '../mini_imagenet/images'
train_path =  '../mini_imagenet/csvsplits/train.csv'
valid_path = '../mini_imagenet/csvsplits/valid.csv'
test_path = '../mini_imagenet/csvsplits/test.csv'
separator = ';'

#*############################ 
#*       Hyperparameters     #
#*############################
n_supports = 5
n_queries = 5
batch_size = 60
learning_rate = 0.01
momentum = 0.9
n_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#*################################
#*     Chargement du dataset     #
#*################################
paths = {'root_dir':dataset_path, 'train_dir':train_path, 'valid_dir':valid_path, 'test_dir':test_path}
transform = transforms.Compose([
                                transforms.Resize((84,84)),
                                transforms.ToTensor()
                               ])
train_set, valid_set, test_set = load_split_datasets(paths, n_supports, n_queries, transforms=transform)

sets = {'train_set':train_set, 'valid_set':valid_set, 'test_set':test_set}
train_loader, valid_loader, test_loader = load_dataloaders(sets, batch_size)

model = ProtoNet()

if use_gpu:
    model = model.cuda()

#*################################
#*   Parametres d'entrainement   #
#*################################
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate,
                             momentum=momentum,
                             nesterov=True,
                             weight_decay=0.01)

criterion = PrototypicalLoss(n_supports)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                  mode='min',
#                                                  patience=patience,
#                                                  verbose=True,
#                                                  threshold=accuracy_threshold)

#*#################################
#*           Entrainement         #
#*#################################
history = History()
# best_loss = 1
for epoch in range(n_epochs):
    progress_bar = tqdm(train_loader, desc="Epoch {}".format(epoch))
    model.train()

    for idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description("Epoch {}".format(epoch))
        
        inputs = torch.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4]))
        targets = torch.reshape(targets, (1, targets.shape[0] * targets.shape[1])).squeeze()

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        predictions = model(inputs)

        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.cpu().data.numpy()})

    # if scheduler:
    #       scheduler.step(val_loss)
    # print('Accuracy: %2f' % val_loss)

    # Deep copy the best model
    # if val_loss < best_loss:
    #     best_loss = val_loss
    #     best_model_weights = copy.deepcopy(model.state_dict())
    #     torch.save(best_model_weights, filename)

    # history.save(train_acc, val_acc, train_loss, val_loss, optimizer.param_groups[0]['lr'])


#*#################################
#*             Tests              #
#*#################################
# # Charger le meilleur modele enregistre
# state_dict = load(save_path)
# net.load_state_dict(state_dict)
# print('Précision en test: {:.2f}'.format(test(net, criterion, test_set, batch_size)))
