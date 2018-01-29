from prototypical_batch_sampler import PrototypicalBatchSampler
from omniglot import Omniglot
from torchvision.models.resnet import resnet152
from torch import nn
import torch
from loss import prototypical_loss as loss
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm

cuda = True
epochs = 10000
iterations = 100
num_query_tr = 5
num_support_tr = 5
num_query_val = 15
num_support_val = 5
classes_per_it_tr = 60
classes_per_it_val = 5

train_dataset = Omniglot(mode='train', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]))

val_dataset = Omniglot(mode='val', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]))

tr_sampler = PrototypicalBatchSampler(train_dataset.y, classes_per_it=classes_per_it_tr,
                                      num_support=num_support_tr, num_query=num_query_tr, iterations=iterations)
tr_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=tr_sampler)

val_sampler = PrototypicalBatchSampler(val_dataset.y, classes_per_it=classes_per_it_val,
                                       num_support=num_support_val, num_query=num_query_val, iterations=iterations)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_sampler=val_sampler)

model = resnet152(pretrained=True)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.conv1.weight.data = model.conv1.weight.data.sum(1).unsqueeze(1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 64)

optim = torch.optim.Adam(model.fc.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

best_acc = 0

if cuda:
    model = model.cuda()


for epoch in range(epochs):
    print('=== Epoch: {} ==='.format(epoch))
    tr_iter = iter(tr_dataloader)
    for batch in tqdm(tr_iter):
        x, y = batch
        x, y = Variable(x), Variable(y)
        if cuda:
            x, y = x.cuda(), y.cuda()
        output = model(x)
        l, acc = loss(output, target=y, n_support=num_support_tr)
        l.backward()
        optim.step()
        train_loss.append(l.data[0])
        train_acc.append(acc.data[0])
    print(np.mean(train_loss[-iterations:]), np.mean(train_acc[-iterations:]))
    val_iter = iter(val_dataloader)
    for batch in tqdm(val_iter):
        x, y = batch
        x, y = Variable(x), Variable(y)
        if cuda:
            x, y = x.cuda(), y.cuda()
        output = model(x)
        l, acc = loss(output, target=y, n_support=num_support_tr)
        val_loss.append(l.data[0])
        val_acc.append(acc.data[0])
    avg_acc = np.mean(val_acc[-iterations:])
    print(np.mean(val_loss[-iterations:]), avg_acc)
    if avg_acc > best_acc:
        print('Best model, Acc: {}'.format(avg_acc))
        torch.save(model.fc.state_dict(), 'best_model.t7')
        best_acc = acc

    lr_scheduler.step()
