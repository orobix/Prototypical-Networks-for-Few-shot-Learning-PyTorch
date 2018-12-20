import numpy
import torch

from protonet import ProtoNet
from utils.few_shot_parameters import FewShotParameters
from utils.meta_test import meta_test
from utils.meta_train import meta_train
from utils.dataloader import load_meta_train_set, load_meta_test_set

use_gpu = torch.cuda.is_available()
paths = {'root_dir': '../mini_imagenet/images',
         'train_dir': '../mini_imagenet/csvsplits/train.csv',
         'valid_dir': '../mini_imagenet/csvsplits/valid.csv',
         'test_dir': '../mini_imagenet/csvsplits/test.csv'}
best_learner_parameters_file = 'best_protonet.pt'
best_learner_grid_search_parameters_file = 'best_protonet_gs.pt'


# Control parameters
EXECUTE_TRAINING = 0
EXECUTE_TEST = 0
GRID_SEARCH = 1


def createModel():
    model = ProtoNet()
    if use_gpu:
        model = model.cuda()
    return model


def getTrainingAndValidationSets(paths):
    train_set, valid_set = load_meta_train_set(paths)
    sets = {'train_set': train_set,
            'valid_set': valid_set}
    return sets

#*#################################
#*             Train              #
#*#################################
if EXECUTE_TRAINING:
    model = createModel()
    meta_train_params = FewShotParameters()

    sets = getTrainingAndValidationSets(paths)

    meta_train_params.set_train_parameters(model, sets)
    best_learner_weights, _ = meta_train(model, meta_train_params, use_gpu)
    torch.save(best_learner_weights, best_learner_parameters_file)

#*#################################
#*             Tests              #
#*#################################
if EXECUTE_TEST:
    model = createModel()
    state_dict = torch.load(best_learner_parameters_file)
    model.load_state_dict(state_dict)
    meta_test_params = FewShotParameters()

    test_set = load_meta_test_set(paths)

    meta_test_params.set_test_parameters(test_set)
    avg_acc = meta_test(model, meta_test_params, use_gpu)
    print('Average test accuracy: {}'.format(avg_acc))


#*#################################
#*             Grid               #
#*#################################
if GRID_SEARCH:
    best_learner_weights = None
    best_valid_acc = 0
    best_lambda = 0
    lambdas = numpy.logspace(-2, 1, 10)

    sets = getTrainingAndValidationSets(paths) # instancié une seule fois

    for l in lambdas:

        model = createModel() # nouveau pour chaque lambda (vraie grid search)
        meta_train_params = FewShotParameters()
        meta_train_params.set_train_parameters(model, sets)

        meta_train_params.l1_lambda = l
        learner_weights, valid_acc = meta_train(model, meta_train_params, use_gpu)
        print('Current lambda %.5f and valid accuracy %.5f' % (l, valid_acc))
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_learner_weights = learner_weights
            best_lambda = l
            torch.save(learner_weights, best_learner_grid_search_parameters_file)

    meta_test_params = FewShotParameters()

    test_set = load_meta_test_set(paths)

    meta_test_params = meta_test_params.set_test_parameters(test_set)
    model = createModel()
    state_dict = torch.load(best_learner_grid_search_parameters_file)
    model.load_state_dict(state_dict)
    test_acc = meta_test(model, meta_test_params, use_gpu)
    print('Test accuracy: {}'.format(test_acc))
    print("Best model validation accuracy: {}".format(best_valid_acc))
    print("Best lambda value: {}".format(best_lambda))

