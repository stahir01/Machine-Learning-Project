import torch
from torch import nn, optim
from data_loading import load_data
from model import NewUNet

from tqdm import tqdm

from matplotlib.pyplot import imshow
from data_visualization import plot_pair

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

def train_model(model, train_loader, optimizer, criterion, device, num_epoch = 10):

   resize = Resize(324, InterpolationMode.NEAREST) # just temporary

   for epoch in range(num_epoch):

      train_loss_values, valid_loss_values = [], []
      for i, data in enumerate(tqdm(train_loader)):
         image, mask = data[0].to(device), data[1].to(device).int()

         optimizer.zero_grad()

         output = model(image)

         mask_ = torch.empty((2,2,512,512))
         mask_[:,1] = mask[:,0]
         mask_[:,0] = ~mask[:,0].int()
         mask_ = resize(mask_).to(device)

         # output.requires_grad = True

         loss = criterion(output[:,0], mask_[:,0])
         loss.backward()
         optimizer.step()
         
         train_loss_values.append(loss.item())
      
      print(f'Average loss: {torch.tensor(train_loss_values).mean()}')

      

def main():
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   print(f"Running on: {device}")

   train_loader, test_loader = load_data()
   model = NewUNet().to(device)

   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   
   criterion = nn.CrossEntropyLoss()

   train_model(model, train_loader, optimizer, criterion, device, num_epoch=10)

if __name__ == "__main__":
   main()