import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision.transforms import ToTensor

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

        return image, label


if __name__ == '__main__':
    dataset = ISBIEMSegDataset('./data/isbi_em_seg', transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=5)
    
    s = next(iter(dataloader))
    a = 1