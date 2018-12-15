import torch
from torch.autograd import Variable

from protonet import ProtoNet
from utils.few_shot_parameters import FewShotParameters
from utils.meta_test import meta_test
from utils.meta_train import meta_train

use_gpu = torch.cuda.is_available()
paths = {'root_dir': '../mini_imagenet/images',
         'train_dir': '../mini_imagenet/csvsplits/train.csv',
         'valid_dir': '../mini_imagenet/csvsplits/valid.csv',
         'test_dir': '../mini_imagenet/csvsplits/test.csv'}
best_learner_parameters_file = 'best_protonet.pt'

# Control parameters
EXECUTE_TRAINING = 1
EXECUTE_TEST = 1

model = ProtoNet()

if use_gpu:
    model = model.cuda()

#*#################################
#*             Train              #
#*#################################
if EXECUTE_TRAINING:
    meta_train_parameters = FewShotParameters(model, 'train', paths)
    best_learner_weights = meta_train(model, meta_train_parameters, use_gpu)
    torch.save(best_learner_weights, best_learner_parameters_file)

#*#################################
#*             Tests              #
#*#################################
if EXECUTE_TEST:
    state_dict = torch.load(best_learner_parameters_file)
    model.load_state_dict(state_dict)
    meta_test_parameters = FewShotParameters(model, 'test', paths)
    avg_acc = meta_test(model, meta_test_parameters, use_gpu)
