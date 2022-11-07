from isbi_em_seg_dataset import *
from model import *

import torch

def train(epochs, batch_size=5, learning_rate=1e-3):
    dataset = ISBIEMSegDataset('./data/isbi_em_seg', transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("****** " + str(device) + " is used. ******")

    criterium = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Learning rate decay exponetially
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        for batch_index, (image, label) in enumerate(dataloader):
            model.train()

            image, label = image.to(device), label.to(device)
        
            optimizer.zero_grad()

            # output = torch.maximum(model(image)[0, 0, :, :], model(image)[0, 1, :, :])
            output = model(image)[0, :, :, :]

            loss = criterium(output, label)
            loss.backward()

            optimizer.step()

            if batch_index % 10 == 0:
                print("Training Epoch: {} [{}/{} ({:.0f}%) \t Loss:{:.6f}]".format(epoch, batch_index*len(image), len(dataloader), 100. * batch_index/len(dataloader), loss.item()), end='\r')
        
        scheduler.step()

if __name__ == "__main__":
    
    train(epochs=2)