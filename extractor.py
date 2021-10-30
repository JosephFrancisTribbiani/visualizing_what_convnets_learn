import torch
import torchvision
from torchvision import transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def extract(model, layers: list, img, loc='./'):
  """
  The function saves outputs of all filters at each layer from layer list.
  Input:
  - model - neural network to be analyzed
  - layers: list of integers - the numbers of the model layers from which we want to get the output
  - img: PIL image as input for neural network
  - loc: str - the location where the outputs will be saved
  Output: None
  """
  data = transforms.Compose([transforms.CenterCrop(min(img.size)),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]), ])(img)
  model.eval()
  with torch.no_grad():
    for i, layer in tqdm(enumerate(layers), total=len(layers)):
      output = get_layer_output(data, model[:layer + 1])
      grid = create_grid(output)
      if not os.path.isdir(loc):
        os.mkdir(loc)
      save_img(loc + f'layer_{layer}.jpg', grid)

def get_layer_output(in_tensor: torch.tensor, submodel):
  """
  The function returns the output of submodel.
  Inputs:
  - in_tensor: torch.tensor - torch tensor wuth size (3, 224, 224)
  - submodel - model from which we want to get the output (croped on top at needed layer)
  Output:
  - numpy array size (num filters at last submodel layer, h, w)
  """
  return submodel(in_tensor.unsqueeze(0)).squeeze(0).detach().numpy()

def create_grid(array):
  """
  The function combine all images from array in one as grig of images
  Input: numpy array size (n filters, h, w)
  Output: numpy array
  """
  i, h, w = array.shape
  ncols = int(np.ceil(np.sqrt(i)))
  nrows = int(i // ncols + (i % ncols > 0))
  n_empty = ncols**2 - i
  empty_array = np.tile(np.full_like(array[-1], np.min(array)), (n_empty, 1, 1))
  array = np.concatenate((array, empty_array), axis=0)

  grid = array.reshape(ncols, ncols, h, w) \
              .swapaxes(1, 2) \
              .reshape(h*ncols, w*ncols)
  return grid

def save_img(loc, grid):
  plt.imsave(loc, grid, cmap='viridis')