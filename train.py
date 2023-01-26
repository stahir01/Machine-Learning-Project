import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time

from data_loading import load_data
from model import NewUNet

from tqdm import tqdm

from matplotlib.pyplot import imshow
from data_visualization import plot_pair

from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt

from metrics import calculate_iou

def get_test_metrics(model, test_loader, criterion, device):
   losses, ious = [], []
   with torch.no_grad():
      for i, data in enumerate(test_loader):

         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)
         output = output[:,1].unsqueeze(dim=1)

         loss = criterion(output, mask)
         losses.append(loss.item())

         iou = calculate_iou(output, mask)
         ious.append(iou.item())

   return torch.tensor(losses).mean(), torch.tensor(ious).mean()

def custom_loss(output, mask):
   # return (output - mask).abs().mean(dim=(1,2,3)).mean()
   return ((output - mask)**2).mean(dim=(1,2,3)).norm()

def test_model(model, test_loader, criterion, device):
    
    model = model.eval()
    predictions, mask_result = [], []
    
    with torch.no_grad():
      for i, data in enumerate(tqdm(test_loader)):
         image, mask = data[0].to(device), data[1].to(device)
         output = model(image)
         
         test_output = model(image)  
         test_output = test_output[:,1].unsqueeze(dim=1)  

         #Test score       
         test_score = calculate_iou_batch(test_output, mask)
         avg_test_score = (torch.tensor(test_score).mean()).cpu().numpy().astype(float)

         #Pixel accuracy
         pixel_score = pixel_accuracy(test_output, mask)
         avg_pixel_score = (torch.tensor(pixel_score).mean()).cpu().numpy().astype(float)

         #Dice Score
         dice_score = dice_coeff(test_output, mask)
         avg_dice_score = (torch.tensor(dice_score).mean()).cpu().numpy().astype(float)


         predictions.append(output)
         mask_result.append(mask) #Used for plotting images labels/masks

         
    predictions = torch.cat(predictions, dim=0)
    mask = torch.cat(mask_result, dim=0)
    return predictions, mask, avg_test_score, avg_pixel_score, avg_dice_score

def train_model(model, train_loader, optimizer, criterion, device, num_epoch = 100):
  
  avg_loss_val, avg_score_val = [], []
  model.train()

  for i in range (1, num_epoch+1):
    
    losses = []
    
    for batch_index, (image, label) in enumerate(tqdm(train_loader)):
      image, mask = image.to(device), label.to(device)
      
      optimizer.zero_grad()
      output = model(image)
      output = output[:,1].unsqueeze(dim=1)
      
      loss = criterion(output, mask)
      score = calculate_iou_batch(output, label)
      losses.append(loss)
      loss.backward()
      optimizer.step()
       
      avg_loss = torch.tensor(losses).mean()
      avg_score = torch.tensor(score).mean()

    avg_loss_val.append(avg_loss)
    avg_score_val.append(avg_score)
    print("Epoch {0}: train_loss {1} \t train_score {2}".format(i, avg_loss, avg_score))
  plt.figure(figsize=(25,5))
  plt.plot(avg_loss_val,'-o')
  plt.plot(avg_score_val,'-x')
  plt.xlabel('epochs')
  plt.ylabel('Avg Loss & Score')
  plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
  plt.legend(['Train Loss', 'Train Score'])
  plt.title('Train Loss & Train Score')
  plt.show()
  plt.pause(5)
  plt.close()








'''
def train_one_epoch(model, train_loader, optimizer, criterion, device):
   losses, ious = [], []
   for i, (image, mask) in enumerate(tqdm(train_loader, leave=False)):
      image, mask = image.to(device), mask.to(device)

      optimizer.zero_grad()

      output = model(image)
      output = output[:,1].unsqueeze(dim=1)

      loss = criterion(output, mask)
      iou = calculate_iou(output, mask)
      losses.append(loss.item())
      ious.append(iou.item())
      loss.backward()
      optimizer.step()

   avg_loss = torch.tensor(losses).mean()
   avg_iou = torch.tensor(ious).mean()
   return avg_loss, avg_iou
         
def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epoch = 100):
   model.train()

   avg_ious_train, avg_ious_test = [], []
   avg_losses_train, avg_losses_test = [], []

   for epoch in range(num_epoch): 
      avg_loss_train, avg_iou_train = train_one_epoch(model, train_loader, optimizer, criterion, device)
      avg_loss_test, avg_iou_test = get_test_metrics(model, test_loader, criterion, device)

      avg_ious_train.append(avg_iou_train)
      avg_ious_test.append(avg_iou_test)
      avg_losses_train.append(avg_loss_train)
      avg_losses_test.append(avg_loss_test)
      print(epoch, avg_loss_train, avg_iou_train, avg_loss_test, avg_iou_test)

   plt.plot(avg_losses_train, label='Training Loss')
   plt.plot(avg_ious_train, label='Training IoU')
   plt.plot(avg_losses_test, label='Testing Loss')
   plt.plot(avg_ious_test, label='Testing IoU')
   plt.legend(loc='best')
   plt.grid(visible=True, which='both')
   plt.show()

def main():
   pass

   # torch.save(model.state_dict(), 'models/model_' + SUFFIX + '.pth')
   # print('models/model_' + SUFFIX + '.pth')

if __name__ == "__main__":
   main()
'''