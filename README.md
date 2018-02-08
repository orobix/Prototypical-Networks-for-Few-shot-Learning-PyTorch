# Prototypical-Networks-for-Few-shot-Learning-PyTorch
Simple Implementation of Prototypical Networks for Few Shot Learning (https://arxiv.org/abs/1703.05175) in Pytorch

# Omniglot dataset class

Kudos to [@ludc](https://github.com/ludc) for his contribute: https://github.com/pytorch/vision/pull/46.
We will use the official dataset when it will be added to torchvision if it doesn't imply big changes to the code.

# Prototypical Batch Sampler

As described in its PyDoc, this class is used to generate the indexes of each batch for a ptrototypical training algorithm.

# Prototypical Loss

Compute the loss as in the cited paper, mostly inspired by [this code](https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py) by one of the authors.


# Training

Please note that the training code is here just for demonstration purposes. 
We use a pretrained resnet model as feature extractor and train only the last layer as classifier.