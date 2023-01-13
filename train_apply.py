import torch
from torch import nn, optim

from data_loading import load_data
from model import NewUNet

from train import train_model, test_model

NUM_EPOCHS = 25
LR = 0.01
MOMENTUM = .99
BATCH_SIZE = 1

SUFFIX = f'EP_{NUM_EPOCHS}_LR_{LR}_MOM_{MOMENTUM}_BS_{BATCH_SIZE}'

def train_apply(method = 'train_model',dataset = 'isbi_em_seg', num_epochs=25, lr=0.01, momentum=0.99, batch_size=3, in_channel = 1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    train_loader, test_loader = load_data(dataset=dataset, batch_size=batch_size, n_train=0.8, n_test=0.2)
    
    # switch to RGB dataset
    if dataset != 'isbi_em_seg':
        in_channel = 3

    model = NewUNet(in_channel).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)   
    criterion = nn.BCEWithLogitsLoss()

    eval(f'{method}(model, train_loader, optimizer, criterion, device, num_epoch={num_epochs})')

    predictions = test_model(model, test_loader, device)

    return predictions

def main():
    predictions = train_apply(num_epochs=5)
    print(predictions.shape)

if __name__ == '__main__':
    main()