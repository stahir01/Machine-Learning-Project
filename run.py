import torch
from torch import nn, optim
from data_loading import load_data
from model import NewUNet

from tqdm import tqdm

from matplotlib.pyplot import imshow
from data_visualization import plot_pair

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt

def train_model(model, train_loader, optimizer, criterion, device, num_epoch = 10):

   train_losses, test_losses = [], []
   for epoch in range(num_epoch):

      train_loss_values, valid_loss_values = [], []
      for i, data in enumerate(tqdm(train_loader)):
         image, mask = data[0].to(device), data[1].to(device)

         optimizer.zero_grad()

         output = model(image)[:,1].unsqueeze(dim=1)

         loss = criterion(output, mask)
         loss.backward()
         optimizer.step()
         
         train_loss_values.append(loss.item())
      
      mean_loss = torch.tensor(train_loss_values).mean()
      print(f'Average loss (@ {epoch+1}): {mean_loss}')
      train_losses.append(mean_loss)
   
   plt.plot(train_losses)
   plt.show()

def main():
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(f"Running on: {device}")

   train_loader, test_loader = load_data(batch_size=3, n_train=.5, n_test=.5)
   model = NewUNet().to(device)
   
   optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.99)   
   criterion = nn.BCEWithLogitsLoss()

   train_model(model, train_loader, optimizer, criterion, device, num_epoch=150)

if __name__ == "__main__":
   main()