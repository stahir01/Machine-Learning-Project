from isbi_em_seg_dataset import *
from model import *

import torch

def train(epochs, batch_size=5, learning_rate=1e-3):
    dataset = ISBIEMSegDataset('./data/isbi_em_seg', transform=Compose([ToTensor(), Resize((572, 572))]))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterium = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    for epoch in epochs:
        for batch_index, (image, label) in enumerate(dataloader):
            model.train()

            image, label = image.to(device), label.to(device)
        
            optimizer.zero_grad()
            output = model(image)

            loss = criterium(output, label)
            loss.backward()

            optimizer.step()

            if batch_index % 10 == 0:
                print("Training Epoch: {} [{}/{} ({:.0f}%) \t Loss:{:.6f}]".format(epoch, batch_index*len(image), len(dataloader), 100. * batch_index/len(dataloader), loss.item()), end='\r')

if __name__ == "__main__":
    
    train(epoch=2)