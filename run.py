import matplotlib.pyplot as plt
from matplotlib.pylab import plt
from matplotlib.ticker import MaxNLocator
from isbi_em_seg_dataset import *
from model import *
from data_transformation import *
from train_validate import *


def train_model(dataset, num_epoch = 10, batch_size = 2, learning_rate = 0.01, weight_decay = 0.001, split_size = 0.8):
   dataset = ISBIEMSegDataset(dataset, transform=ToTensor())
   #train_data, validate_data = split_data(dataset, split_size)
   #print(len(train_data), len(validate_data))

   train_loss_values, valid_loss_values = [], []

   for i in range(1, num_epoch+1):
     train_loss = train(dataset, batch_size, learning_rate, weight_decay)
     #valid_loss = validate(validate_data, batch_size, learning_rate, weight_decay)
     
     train_loss_values.append(train_loss)
     #valid_loss_values.append(valid_loss)
     print("Epoch {0}: train_loss {1}".format(i, train_loss))

    #print result
   fig, ax = plt.subplots(1, 1)
   plt.plot(train_loss_values, color = 'red', label = "Training Loss")
   #plt.plot(valid_loss_values, color = 'blue', label = "Validation Loss")
   plt.yticks(np.arange(0, 1, step=0.2))
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend()
   plt.plot()



if __name__ == "__main__":
    train_model(dataset='/Users/syedalimuradtahir/Documents/WS 2022/Machine Learning Project/Machine-Learning-Project/data/isbi_em_seg')
