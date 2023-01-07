import torch
from torch import nn, optim
import torch.nn.functional as F

from data_loading import load_data
from model import NewUNet

from tqdm import tqdm

from matplotlib.pyplot import imshow
from data_visualization import plot_pair

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt

import elasticdeform

def get_test_loss(model, test_loader, criterion, device):
   losses = []
   with torch.no_grad():
      for i, data in enumerate(tqdm(test_loader)):

         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)
         # output = F.softmax(output, dim=1)
         output = output[:,1].unsqueeze(dim=1)

         loss = criterion(output, mask)
         losses.append(loss.item())
   return torch.tensor(losses).mean()

def custom_loss(output, mask):
   # return (output - mask).abs().mean(dim=(1,2,3)).mean()
   return ((output - mask)**2).mean(dim=(1,2,3)).norm()

def test_model(model, test_loader, device):
   model.eval()

   predictions = []
   with torch.no_grad():
      for i, data in enumerate(tqdm(test_loader)):
         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)
         predictions.append(output)

   predictions = torch.cat(predictions, dim=0)
   return predictions

def train_one_epoch(model, train_loader, optimizer, criterion, device):
   losses = []
   for i, data in enumerate(tqdm(train_loader, leave=False)):
      # elastic deform
      deformed_images = []
      deformed_masks = []
      for image, mask in zip(data[0], data[1]):
         [image_deformed, mask_deformed] = elasticdeform.deform_random_grid([image.numpy(), mask.numpy()], axis=(1, 2), sigma=25, points=3, rotate=30, zoom=1.5)
         deformed_images.append(image_deformed)
         deformed_masks.append(mask_deformed)
      deformed_images = torch.tensor(deformed_images)
      deformed_masks = torch.tensor(deformed_masks)

      image, mask = image.to(device), mask.to(device)

      optimizer.zero_grad()

      output = model(image)
      output = output[:,1].unsqueeze(dim=1)

      loss = criterion(output, mask)
      losses.append(loss)
      loss.backward()
      optimizer.step()

   avg_loss = torch.tensor(losses).mean()
   return avg_loss 
         
def train_model(model, train_loader, optimizer, criterion, device, num_epoch = 100):
   model.train()

   pbar = tqdm(range(num_epoch))
   pbar.set_description(f'Loss: _.____')
   for epoch in pbar: 
      avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

      pbar.set_description(f'Loss: {avg_loss:0.4}')

def main():
   pass

   # torch.save(model.state_dict(), 'models/model_' + SUFFIX + '.pth')
   # print('models/model_' + SUFFIX + '.pth')

if __name__ == "__main__":
   main()