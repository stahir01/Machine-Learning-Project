import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose

class ISBIEMSegDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform

        self.path_images = os.path.join(self.root_dir, 'images')
        self.path_labels = os.path.join(self.root_dir, 'labels')

        self.image_paths = os.listdir(self.path_images)
        self.label_paths = os.listdir(self.path_labels)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = io.imread(os.path.join(self.path_images, self.image_paths[index])) 
        label = io.imread(os.path.join(self.path_labels, self.label_paths[index])) 

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image.float(), label.float()

def load_data(dataset='isbi_em_seg', transformation=None, n_train=None, n_test=None, batch_size=2):

    ds_reg = [
        'isbi_em_seg',
    ]

    if not dataset in ds_reg:
        print(f'Dataset not in registry, available datasets are: {ds_reg}')
    else:
        if not transformation:
            transformation = ToTensor()
        else:
            transformation = torch.nn.Sequential(
                ToTensor(),
                transformation
            )

        dataset = ISBIEMSegDataset(f'./data/{dataset}', transform=transformation)  

        if not (n_train or n_test):
            total_len = len(dataset)
            n_train = int(.8 * total_len)
            n_test = total_len - n_train

        train_set, test_set = random_split(dataset=dataset, lengths=[n_train, n_test]) 

        return DataLoader(train_set, batch_size=batch_size), DataLoader(test_set, batch_size=batch_size)       

if __name__ == '__main__':
    
    train_set, test_set = load_data()
    a=1