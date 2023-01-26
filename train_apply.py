import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from data_loading import load_data
from model import NewUNet

from train import train_model, test_model
from entropy_loss import DiceBCELoss

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
    if dataset != 'isbi_em_seg' and dataset != 'isbi_em_seg_100':
        in_channel = 3

    model = NewUNet(in_channel).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)   
    #criterion = nn.BCEWithLogitsLoss()
    criterion = DiceBCELoss()

    eval(f'{method}(model, train_loader, test_loader, optimizer, criterion, device, num_epoch={num_epochs})')

    predictions, mask, avg_test_score, avg_pixel_score, avg_dice_score = test_model(model, test_loader, criterion, device)

    return predictions

def main():
    predictions = train_apply(num_epochs=50)
    print(predictions.shape)

if __name__ == '__main__':
    predictions, mask, test_score, pixel_score, dice_score = train_apply(num_epochs=NUM_EPOCHS, lr=LR, batch_size=BATCH_SIZE, dataset='isbi_em_seg')
    #print(predictions.shape)

    prediction_test = predictions#[:, 1:2, :, :]
    mask_test = mask
    #Final Result
    fig = plt.figure(figsize=(20,20))
    for i in range (len(prediction_test)):
        while i < 5:
            predict_results = prediction_test[i].cpu().numpy()
            mask_results = mask_test[i].cpu().numpy()


            #predict_results = (predict_results * 255.0).astype("uint8")
            #mask_results = (mask_results * 255.0).astype("uint8")

            predict_results = predict_results[0]
            mask_results = mask_results[0]

            predict_results = predict_results.squeeze()
            mask_results = mask_results.squeeze()
            #print("Predict Results: ", predict_results)
            #print("Mask Results: ", mask_results)

            plt.subplot(5,2,2*i+1)
            plt.imshow(mask_results)
            plt.axis("off")
            plt.subplot(5,2,2*i+2)
            plt.imshow(predict_results, cmap = 'gray') 
            plt.axis("off")
            i+=1
    plt.show()