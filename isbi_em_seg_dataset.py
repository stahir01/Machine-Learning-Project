import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from data_visualization import show

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
        
        print(image.shape)
        print(label.shape)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label


if __name__ == '__main__':
    dataset = ISBIEMSegDataset('./data/isbi_em_seg', transform=ToTensor())
    
    BATCH_SIZE = 5
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    s = next(iter(dataloader))
    # a = 1
    
    # [5, 1, 572, 572]
    print(s[0].shape)
    examples = [np.array(s[0][i])[0,:,:] for i in range((s[0].shape)[0])]
    
    # Adjust height and width for visualization
    show(examples, num_img=BATCH_SIZE, height=1, width=5)