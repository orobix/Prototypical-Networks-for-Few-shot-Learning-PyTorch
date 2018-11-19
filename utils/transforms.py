import torch
from torchvision import transforms
import numpy as np
import pdb

class ToTensor(object):
    """ Convert the input data to Tensors."""
    def __call__(self, sample):
        img, speed = sample[0], sample[1]
        img = torch.from_pil_image(img)
        speed = torch.from_numpy(np.array(speed))
        return [img, speed]
