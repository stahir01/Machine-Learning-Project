import matplotlib.pyplot as plt
import numpy as np
import os

def show(x, outfile = False, figsize_=(20,20), num_img=25, height=5, width=5):
    """
    Data visualization of test data

    Args:
        x (numpy array): Test dataset
        outfile (boolean, optional): Specify wether storing generated visualization is needed. Defaults to False.
        figsize_ (tuple, optional): Size of generated figure. Defaults to (20,20).
        num_img (int, optional): Number of images containing in the figure. Defaults to 25.
        height (int, optional): Number of rows. Defaults to 5.
        width (int, optional): Number of colums. Defaults to 5.
    """    

    plt.figure(figsize=figsize_)
    for i in range(num_img):
        plt.subplot(height,width,i+1)  
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
    
    if outfile == True:
        # Check if the figure folder exists
        if not os.path.exists(os.getcwd()+"/figure"):
            os.makedirs(os.getcwd()+"/figure")
        plt.savefig("./figure/data_visualization.png")
        
    plt.show()


if __name__ == "__main__":
    X_test = np.load("./data/X_test.npy")
    show(X_test, True)
    