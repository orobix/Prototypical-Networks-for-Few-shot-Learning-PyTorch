# Prototypical-Networks-for-Few-shot-Learning-PyTorch
Simple Implementation of Prototypical Networks for Few Shot Learning (https://arxiv.org/abs/1703.05175) in Pytorch

# Prototypiacl Networks

As shown in the reference paper Prototypical Networks are trained to embed samples features in a vectorial space, in particular, at each episode (iteration), a number of samples for a subset of classes are selected and sent thourgh the model, for each subset of class 'c' a number of samples'features ('n_support') are used to guess the prototype (their baricentre coordinates in the vactorial space) for that class, so then the distances between the remaining 'n_query' samples and their class baricentre can be minimized.

![Prototypical Networks](doc/imgs/proto-1.png)

# Omniglot dataset

Kudos to [@ludc](https://github.com/ludc) for his contribute: https://github.com/pytorch/vision/pull/46.
We will use the official dataset when it will be added to torchvision if it doesn't imply big changes to the code.

# Prototypical Batch Sampler

As described in its PyDoc, this class is used to generate the indexes of each batch for a ptrototypical training algorithm.

# Prototypical Loss

Compute the loss as in the cited paper, mostly inspired by [this code](https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py) by one of its authors.


# Training

Please note that the training code is here just for demonstration purposes. It use a pretrained resnet model as feature extractor and train only the last layer as a classifier.