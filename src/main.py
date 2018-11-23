import math

import torch
import copy
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from protonet import ProtoNet
from prototypical_loss import PrototypicalLoss
from torchvision import transforms
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
filename = 'best_protonet.pt'

#*############################ 
#*       Hyperparameters     #
#*############################
n_support = 5
n_query = 5
n_samples_per_class = n_support + n_query
learning_rate = 1e-3
n_epochs = 600
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_episodes = 100
classes_per_it = (40, n_samples_per_class, n_samples_per_class)

#*################################
#*     Chargement du dataset     #
#*################################
paths = {'root_dir': dataset_path, 'train_dir': train_path, 'valid_dir': valid_path, 'test_dir': test_path}
transform = transforms.Compose([
                                transforms.Resize((84, 84)),
                                transforms.ToTensor()
                               ])
train_set, valid_set, test_set = load_split_datasets(paths, n_support, n_query, transforms=transform)

sets = {'train_set': train_set, 'valid_set': valid_set, 'test_set': test_set}
train_loader, valid_loader, test_loader = load_dataloaders(sets,
                                                           samples_per_class=n_samples_per_class,
                                                           n_episodes=n_episodes,
                                                           classes_per_it=classes_per_it)

model = ProtoNet()

if use_gpu:
    model = model.cuda()

#*################################
#*   Parametres d'entrainement   #
#*################################
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate)

criterion = PrototypicalLoss(n_support)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                  mode='min',
#                                                  patience=patience,
#                                                  verbose=True,
#                                                  threshold=accuracy_threshold)

#*#################################
#*           Entrainement         #
#*#################################
history = History()
best_model_weights = None
best_avg_acc = 0
for epoch in range(n_epochs):
    progress_bar = tqdm(train_loader, desc="Epoch {}".format(epoch))
    progress_bar.set_description("Epoch {}".format(epoch))

    model.train()
    avg_acc = 0
    for idx, (inputs, targets) in enumerate(progress_bar):

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        predictions = model(inputs)

        loss, train_accuracy = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        avg_acc += train_accuracy.cpu().data.numpy() / n_episodes
        progress_bar.set_postfix({'loss': loss.cpu().data.numpy(), 'acc':avg_acc})

    # if scheduler:
    #       scheduler.step(val_loss)

    # Deep copy the best model
    if avg_acc > best_avg_acc:
         best_avg_acc = avg_acc
         best_model_weights = copy.deepcopy(model.state_dict())

print("Training ended. Saving the best model.")
torch.save(best_model_weights, filename)
print("Best model saved.")

    # history.save(train_acc, val_acc, train_loss, val_loss, optimizer.param_groups[0]['lr'])


#*#################################
#*             Tests              #
#*#################################
# # Charger le meilleur modele enregistre
# state_dict = load(save_path)
# net.load_state_dict(state_dict)
# print('Précision en test: {:.2f}'.format(test(net, criterion, test_set, batch_size)))
