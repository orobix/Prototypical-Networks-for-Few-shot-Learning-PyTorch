import torch
from torch.autograd import Variable

def test(model, test_loader, criterion, n_episodes, use_gpu):

  model.eval()

  avg_acc = 0
  for idx, (inputs, targets) in enumerate(test_loader):

      if use_gpu:
          inputs = inputs.cuda()
          targets = targets.cuda()

      inputs, targets = Variable(inputs), Variable(targets)
      predictions = model(inputs)

      loss, test_accuracy = criterion(predictions, targets)

      avg_acc += test_accuracy.cpu().data.numpy() / n_episodes

  return avg_acc