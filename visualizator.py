import torch
from torch.optim.optimizer import Optimizer

import numpy as np
from tqdm.auto import tqdm


class CustomOptimizer(Optimizer):
  def __init__(self, params, lr=10):
    defaults = dict(lr=lr)
    super().__init__(params, defaults)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        grad /= torch.sqrt(torch.mean(torch.square(grad))) + 1e-5
        state = self.state[p]
        
        if len(state) == 0:
          state['step'] = 0

        p.data.add_(grad, alpha=group['lr'])
        state['step'] += 1
    return loss


def compute_loss(feature_extractor, input_image_, device, filter_index=0):
  activation = feature_extractor(input_image_.unsqueeze(0).to(device))
  filter_activation = activation[:, filter_index, 2:-2, 2:-2]
  return filter_activation.mean()


def visualize_filter(submodel, 
                     layer,
                     input_image,
                     device, 
                     epochs=20,
                     filter_index=0,
                     learning_rate=10):
  running_loss = list()
  input_image_ = input_image.clone().detach().requires_grad_(True)

  feature_extractor = submodel[:layer + 1]
  feature_extractor.to(device)

  optim = CustomOptimizer(params=[input_image_], lr=learning_rate)
    
  for epoch in tqdm(range(epochs)):
    optim.zero_grad()
    loss = compute_loss(feature_extractor, input_image_, device, filter_index)
    loss.backward()
    optim.step()
    running_loss.append(loss.item())
  input_image_ = deprocess_image(input_image_)
  return input_image_, running_loss

def deprocess_image(input_image_):
  input_image_ = input_image_.detach().numpy()
  input_image_ -= input_image_.mean()
  input_image_ /= input_image_.std() + 1e-5
  input_image_ *= 0.15

  # Center crop
  input_image_ = input_image_[:, 25:-25, 25:-25]

  # Clip to [0, 1]
  input_image_ += 0.5
  input_image_ = np.clip(input_image_, 0, 1)

  # Convert to RGB array
  input_image_ *= 255
  input_image_ = np.clip(input_image_, 0, 255).astype("uint8")
  return input_image_