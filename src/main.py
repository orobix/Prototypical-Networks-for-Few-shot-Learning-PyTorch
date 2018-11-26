import torch
import copy
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from protonet import ProtoNet
from prototypical_loss import PrototypicalLoss
from torchvision import transforms
from utils.dataloading import load_dataloaders, load_split_datasets, load_test_set, load_test_dataloaders

from utils.test import test

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

EXECUTE_TRAINING = 1

#*############################ 
#*       Hyperparameters     #
#*############################
n_support = 5
n_query = 5
n_samples_per_class = n_support + n_query
learning_rate = 1e-3
n_epochs = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_episodes = 100
classes_per_it = (32, 8, 8)

#*################################
#*     Chargement du dataset     #
#*################################
paths = {'root_dir': dataset_path, 'train_dir': train_path, 'valid_dir': valid_path, 'test_dir': test_path}
transform = transforms.Compose([
                                transforms.Resize((84, 84)),
                                transforms.ToTensor()
                               ])

if EXECUTE_TRAINING:                               
    train_set, valid_set, test_set = load_split_datasets(paths, n_support, n_query, transforms=transform)

    sets = {'train_set': train_set, 'valid_set': valid_set, 'test_set': test_set}
    train_loader, valid_loader, test_loader = load_dataloaders(sets,
                                                            samples_per_class=n_samples_per_class,
                                                            n_episodes=n_episodes,
                                                            classes_per_it=classes_per_it)
else:
    test_set = load_test_set(paths, n_support, n_query, transforms=transform)
    test_loader = load_test_dataloaders(test_set,
                                    samples_per_class=n_samples_per_class,
                                    n_episodes=n_episodes,
                                    classes_per_it=8)

model = ProtoNet()

if use_gpu:
    model = model.cuda()

#*################################
#*   Parametres d'entrainement   #
#*################################
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate)

criterion = PrototypicalLoss(n_support)

#* Reduce the learning rate by half every 2000 episodes
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=20, # 20 epochs of 100 episodes
                                      gamma=0.5)

#*#################################
#*           Entrainement         #
#*#################################
if EXECUTE_TRAINING:
    PATIENCE_LIMIT = 10
    best_model_weights = None
    best_avg_acc = 0
    patience_counter = 0
    for epoch in range(n_epochs):
        progress_bar = tqdm(train_loader, desc="Epoch {}".format(epoch))
        progress_bar.set_description("Epoch {}".format(epoch))
        model.train()

        if scheduler:
            scheduler.step()

        avg_train_acc = 0
        avg_val_acc = 0
        for (idx, train_batch), (val_idx, val_batch) in zip(enumerate(progress_bar), enumerate(valid_loader)):

            inputs, targets = train_batch
            val_inputs, val_targets = val_batch

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
                val_inputs = val_inputs.cuda()
                val_targets = val_targets.cuda()

            optimizer.zero_grad()

            inputs, targets = Variable(inputs), Variable(targets)
            predictions = model(inputs)

            loss, train_accuracy = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            avg_train_acc += train_accuracy.cpu().data.numpy() / n_episodes

            val_inputs, val_targets = Variable(val_inputs), Variable(val_targets)
            val_predictions = model(val_inputs)

            _, val_acc = criterion(val_predictions, val_targets)
            avg_val_acc += val_acc.cpu().data.numpy() / n_episodes

            progress_bar.set_postfix({'loss': loss.cpu().data.numpy(), 't_acc':avg_train_acc, 'v_acc':avg_val_acc})

        # Deep copy the best model
        if avg_val_acc > best_avg_acc:
            patience_counter = 0
            best_avg_acc = avg_val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE_LIMIT:
            print("Training ended. Saving the best model.")
            torch.save(best_model_weights, filename)
            print("Best model saved.")
            break

    print("Training ended. Saving the best model.")
    torch.save(best_model_weights, filename)
    print("Best model saved.")

#*#################################
#*             Tests              #
#*#################################
# Charger le meilleur modele enregistre
state_dict = torch.load(filename)
model.load_state_dict(state_dict)

avg_acc = test(model, test_loader, criterion, 600, n_episodes, use_gpu)

print('Précision en test: {:.2f}'.format(avg_acc))
