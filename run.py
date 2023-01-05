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

NUM_EPOCHS = 25
LR = 0.01
MOMENTUM = .99
BATCH_SIZE = 3

SUFFIX = f'EP_{NUM_EPOCHS}_LR_{LR}_MOM_{MOMENTUM}_BS_{BATCH_SIZE}'

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

def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epoch = 10):

   train_losses, test_losses = [], []
   for epoch in range(NUM_EPOCHS):

      train_loss_values = []
      for i, data in enumerate(tqdm(train_loader)):
         image, mask = data[0].to(device), data[1].to(device)

         optimizer.zero_grad()

         output = model(image)
         # output = F.softmax(output, dim=1)
         output = output[:,1].unsqueeze(dim=1)

         loss = criterion(output, mask)
         loss.backward()
         optimizer.step()
         
         train_loss_values.append(loss.item())
      
      mean_loss = torch.tensor(train_loss_values).mean()
      print(f'Average loss (@ {epoch+1}): {mean_loss}')
      train_losses.append(mean_loss)
      test_losses.append(get_test_loss(model, test_loader, criterion, device))
   
   plt.plot(train_losses)
   plt.plot(test_losses)
   plt.show()

def main():
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(f"Running on: {device}")

   train_loader, test_loader = load_data(batch_size=BATCH_SIZE, n_train=0.8, n_test=0.2)
   model = NewUNet().to(device)
   
   optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)   
   # criterion = nn.CrossEntropyLoss()
   criterion = custom_loss

   train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epoch=100)

   torch.save(model.state_dict(), 'models/model_' + SUFFIX + '.pth')
   print('models/model_' + SUFFIX + '.pth')

if __name__ == "__main__":
   main()