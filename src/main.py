import numpy
import torch
import os

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
EXECUTE_TRAINING = 0
EXECUTE_TEST = 0
GRID_SEARCH = 1


def createModel():
    model = ProtoNet()
    if use_gpu:
        model = model.cuda()
    return model

#*#################################
#*             Train              #
#*#################################
if EXECUTE_TRAINING:
    model = createModel()
    meta_train_params = FewShotParameters()
    meta_train_params.get_train_parameters(model, paths)
    best_learner_weights = meta_train(model, meta_train_params, use_gpu)
    torch.save(best_learner_weights, best_learner_parameters_file)

#*#################################
#*             Tests              #
#*#################################
if EXECUTE_TEST:
    model = createModel()
    state_dict = torch.load(best_learner_parameters_file)
    model.load_state_dict(state_dict)
    meta_test_params = FewShotParameters()
    meta_test_params.get_test_parameters(paths)
    avg_acc = meta_test(model, meta_test_params, use_gpu)
    print('Average test accuracy: {}'.format(avg_acc))


#*#################################
#*             Grid               #
#*#################################
if GRID_SEARCH:
    best_test_acc = 0
    best_lambda = 0
    lambdas = numpy.logspace(-2, 1, 10)
    meta_train_params = FewShotParameters()
    meta_train_params.get_train_parameters(model, paths)
    meta_test_params = FewShotParameters()
    meta_test_params = meta_test_params.get_test_parameters(paths)

    for l in lambdas:
        model = createModel()
        meta_train_params.l1_lambda = l
        best_learner_weights = meta_train(model, meta_train_params, use_gpu)
        torch.save(best_learner_weights, "./grid_search/model_lambda_{:10.4f}.pt".format(l))

    avg_acc = meta_test(model, meta_test_params, use_gpu)
    print('moyenne des accuracy en test: {} avec lambda={}'.format(avg_acc, l))
    if best_test_acc < avg_acc:
        best_lambda = l

    print("Meilleure valeur de lambda: {}".format(best_lambda))

