import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *





def train(dataset, batch_size, learning_rate, weight_decay):
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # print("The device used:", str(device))

  #Build Model
  model = build_model().to(device)

  #criterion = DiceBCELoss()
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=0.99)
  
  train_loss, target_count = 0, 0
  model.train()
  
  for batch_index, (image, label) in enumerate(train_loader):
    image, label = image.to(device), label.to(device)
    
    #Forward
    output = model(image)[0, :, :, :] 
    loss = criterion(output, label[:, 0, :, :])
    train_loss += loss.item()
    target_count = target_count + label.shape[0]
      

    #Back propgation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return train_loss/target_count


def validate(dataset, batch_size, learning_rate, weight_decay):
  validate_loader = DataLoader(dataset, batch_size=batch_size)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #print("The device used:", str(device))

  #Build Model
  model = build_model().to(device)

  #criterion = DiceBCELoss()
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=0.99)
  
  val_loss, target_count = 0, 0
  model.train()
  
  for batch_index, (image, label) in enumerate(validate_loader):
    image, label = image.to(device), label.to(device)
    
    #Forward
    output = model(image)[0, :, :, :] 
    loss = criterion(output, label[:, 0, :, :])
    val_loss += loss.item()
    target_count = target_count + label.shape[0]
      

    return val_loss/target_count



