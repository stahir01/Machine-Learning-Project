import torch
from torch import nn, optim

from data_loading import load_data
from model import UNet

from tqdm import tqdm

def train_model(model, train_loader, optimizer, criterion, device, num_epoch = 10):
   for epoch in range(num_epoch):

      train_loss_values, valid_loss_values = [], []
      for i, data in enumerate(tqdm(train_loader)):
         input, seg_mask = data[0].to(device), data[1].to(device)

         optimizer.zero_grad()

         output = model(input)
         output = torch.argmax(output, dim=1, keepdim=True).float()
         output.requires_grad = True

         loss = criterion(output, seg_mask)
         loss.backward()
         optimizer.step()
         
         train_loss_values.append(loss.item())
      
      print(f'Average loss: {np.array(train_loss_values).mean()}')

      

def main():
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   train_loader, test_loader = load_data()
   model = UNet().to(device)

   optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   
   criterion = nn.CrossEntropyLoss()

   train_model(model, train_loader, optimizer, criterion, device, num_epoch=1)

if __name__ == "__main__":
   main()