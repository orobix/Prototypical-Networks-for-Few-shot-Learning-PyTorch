# encodig: utf8
from prototypical_batch_sampler import PrototypicalBatchSampler
from omniglot_dataset import OmniglotDataset
from torchvision.models.resnet import resnet18 as resnet
from torch import nn
import torch
import torch.backends.cudnn as cudnn
from prototypical_loss import prototypical_loss as loss
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from parser import get_parser


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    cudnn.benchmark = True


def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    train_dataset = OmniglotDataset(mode='train',
                                    root=opt.dataset_root,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    ]))

    val_dataset = OmniglotDataset(mode='val',
                                  root=opt.dataset_root,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                  ]))

    tr_sampler = PrototypicalBatchSampler(labels=train_dataset.y,
                                          classes_per_it=opt.classes_per_it_tr,
                                          num_support=opt.num_support_tr,
                                          num_query=opt.num_query_tr,
                                          iterations=opt.iterations)

    val_sampler = PrototypicalBatchSampler(labels=val_dataset.y,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_support=opt.num_support_val,
                                           num_query=opt.num_query_val,
                                           iterations=opt.iterations)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler)
    return tr_dataloader, val_dataloader


def init_model(opt):
    '''
    Initialize the pre-trained resnet
    '''
    model = resnet(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.conv1.weight.data = model.conv1.weight.data.sum(1).unsqueeze(1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 64)
    model = model.cuda() if opt.cuda else model
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(model.fc.parameters(), lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optim,
                                           step_size=opt.lr_scheduler_step,
                                           gamma=opt.lr_scheduler_gamma)


def train(opt, tr_dataloader, val_dataloader, model, optim, lr_scheduler):
    '''
    Train the model with the prototypical learning algorithm
    '''
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        for batch in tqdm(tr_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            l.backward()
            optim.step()
            train_loss.append(l.data[0])
            train_acc.append(acc.data[0])
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Train Loss: {}, Train Acc: {}'.format(avg_loss, avg_acc))
        val_iter = iter(val_dataloader)
        for batch in tqdm(val_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            val_loss.append(l.data[0])
            val_acc.append(acc.data[0])
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc > best_acc else ''
        print('Val Loss: {}, Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc > best_acc:
            torch.save(model.fc.state_dict(), '../best_model.pth')
            best_acc = avg_acc

        lr_scheduler.step()

    return best_acc, train_loss, train_acc, val_loss, val_acc


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, val_dataloader = init_dataset(options)
    model = init_model(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
          tr_dataloader=tr_dataloader,
          val_dataloader=val_dataloader,
          model=model,
          optim=optim,
          lr_scheduler=lr_scheduler)


if __name__ == '__main__':
    main()
