import numpy
import torch

from protonet import ProtoNet
from utils.few_shot_parameters import FewShotParameters
from utils.meta_test import meta_test
from utils.meta_train import meta_train
from utils.dataloader import load_meta_train_set, load_meta_test_set, get_training_and_validation_sets

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
PROGRESSIVE_REGULARIZATON = 1


def createModel():
    model = ProtoNet()
    if use_gpu:
        model = model.cuda()
    return model


if EXECUTE_TRAINING:
    model = createModel()
    meta_train_params = FewShotParameters()
    sets = get_training_and_validation_sets(paths)
    meta_train_params.set_train_parameters(model, sets)
    best_learner_weights, _ = meta_train(model, meta_train_params, use_gpu)
    torch.save(best_learner_weights, best_learner_parameters_file)


if EXECUTE_TEST:
    model = createModel()
    state_dict = torch.load(best_learner_parameters_file)
    model.load_state_dict(state_dict)
    meta_test_params = FewShotParameters()

    test_set = load_meta_test_set(paths)

    meta_test_params.set_test_parameters(test_set)
    avg_test_acc, test_std = meta_test(model, meta_test_params, use_gpu)
    print('Average test accuracy: {} with a std of {}'.format(avg_test_acc * 100, test_std * 100))


if PROGRESSIVE_REGULARIZATON:
    best_learner_weights = None
    best_valid_acc = 0
    applied_lambdas = []
    
    lambdas = [0] # Start the training without any regularization
    lambdas.extend(numpy.logspace(-2, 1, 10))

    sets = get_training_and_validation_sets(paths) # instancié une seule fois

    model = createModel() # nouveau pour chaque lambda (vraie grid search)
    meta_train_params = FewShotParameters()
    meta_train_params.set_train_parameters(model, sets)
    for idx, l in enumerate(lambdas):
        meta_train_params.l1_lambda = l
        learner_weights, valid_acc = meta_train(model, meta_train_params, use_gpu)

        if idx == 0:
            torch.save(learner_weights, best_learner_parameters_file)

        print('Current lambda %.5f and valid accuracy %.5f' % (l, valid_acc * 100))
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_learner_weights = learner_weights
            applied_lambdas.append(l)
            torch.save(learner_weights, best_learner_grid_search_parameters_file)

    meta_test_params = FewShotParameters()
    test_set = load_meta_test_set(paths)
    meta_test_params.set_test_parameters(test_set)
    model = createModel()
    state_dict = torch.load(best_learner_grid_search_parameters_file)
    model.load_state_dict(state_dict)
    test_acc, test_std = meta_test(model, meta_test_params, use_gpu)
    print('Test accuracy: {} with a std of {}'.format(test_acc * 100, test_std * 100))
    print("Best model validation accuracy: {}".format(best_valid_acc * 100))
    print("Applied lambda values (in order): {}".format(str(applied_lambdas)))