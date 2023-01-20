import numpy as np
import torch

def calculate_iou(y_pred,y):
    inputs = y_pred.reshape(-1)
    targets = y.reshape(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    smooth = 1    
    iou = (intersection + smooth)/(union + smooth)
    return iou



def calculate_iou_batch(y_pred,y):
    ious = []
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.clone().cpu().detach().numpy()
    y = y.clone().cpu().detach().numpy() 
    
    for pred, label in zip(y_pred, y):
        ious.append(calculate_iou(pred, label))
    iou = np.nanmean(ious)
    return iou 


def dice_coeff(y_pred, y):
  #Flatten them
  inputs = y_pred.reshape(-1)
  targets = y.reshape(-1)
  smooth = 1
  intersection = (y_pred * y).sum()
  union = y_pred.sum() + y.sum()
  dice = (2 * intersection + smooth)/(union + smooth)
  return dice.item()



def pixel_accuracy(y_pred, y):
    # Convert the predictions and labels to boolean tensors
    correct_predictions = torch.eq(y_pred, y).float()
    # Sum the number of correct predictions
    sum_correct = torch.sum(correct_predictions).float()
    # Calculate the total number of pixels in the predictions and labels
    total_pixels = y_pred.numel()
    # Calculate the pixel accuracy by dividing the number of correct predictions by the total number of pixels
    pixel_accuracy = sum_correct / total_pixels
    return pixel_accuracy