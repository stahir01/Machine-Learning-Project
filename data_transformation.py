from torch.utils.data import random_split
from math import ceil, floor
from torchvision import transforms
from isbi_em_seg_dataset import *




def split_data(dataset, split_size = 0.8):
  train_len = int(ceil(split_size*len(dataset)))
  valid_len = len(dataset) - train_len

  train_data, valid_data = random_split(dataset,[train_len, valid_len])

  return train_data, valid_data


def data_augmentation():
  transform = transforms.Compose(
    [
        transforms.RandomCrop(64,64),
        transforms.RandomAutocontrast(p=0.5),
        transforms.Normalize(mean=[0.595425, 0.3518577, 0.3225522], std=[0.19303136, 0.12492529, 0.10577361]),
        transforms.ToTensor(),
    
    ]

)
  
  return transform

if __name__ == "__main__":
    dataset = ISBIEMSegDataset('/Users/syedalimuradtahir/Documents/WS 2022/Machine Learning Project/Machine-Learning-Project/data/isbi_em_seg', transform=ToTensor())
    train_data, valid_data = split_data(dataset, 0.8)
    print(len(train_data), len(valid_data))




