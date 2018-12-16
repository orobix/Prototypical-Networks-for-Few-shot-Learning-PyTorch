import numpy
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
GRID_SEARCH = 0


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
    meta_train_parameters = FewShotParameters(model, 'train', paths)
    best_learner_weights = meta_train(model, meta_train_parameters, use_gpu)
    torch.save(best_learner_weights, best_learner_parameters_file)

#*#################################
#*             Tests              #
#*#################################
if EXECUTE_TEST:
    model = createModel()
    state_dict = torch.load(best_learner_parameters_file)
    model.load_state_dict(state_dict)
    meta_test_parameters = FewShotParameters(model, 'test', paths)
    avg_acc = meta_test(model, meta_test_parameters, use_gpu)
    print('moyenne des accuracy en test: {}'.format(avg_acc))


#*#################################
#*             Grid               #
#*#################################
if GRID_SEARCH:
    best_test_acc = 0
    best_lambda = 0
    lambdas = numpy.logspace(-2, 1, 10)
    for l in lambdas:
        model = createModel()
        meta_train_parameters = FewShotParameters(model, 'train', paths)
        meta_train_parameters.l1_lambda = l
        best_learner_weights = meta_train(model, meta_train_parameters, use_gpu)
        torch.save(best_learner_weights, "grid_search_lambda{}".format(l))

        meta_test_parameters = FewShotParameters(model, 'test', paths)
        avg_acc = meta_test(model, meta_test_parameters, use_gpu)
        print('moyenne des accuracy en test: {} avec lambda={}'.format(avg_acc, l))
        if best_test_acc < avg_acc:
            best_lambda = l

    print("Meilleure valeur de lambda: {}".format(best_lambda))

