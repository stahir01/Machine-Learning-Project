import os
import numpy as np
from sklearn.model_selection import train_test_split
from model import build_model
import tensorflow as tf
from matplotlib import pyplot as plt

def split_data(test_size=0.2):
    """
    Split the data into training and validation data

    Args:
        test_size (float, optional): train/test ratio. Defaults to 0.2.

    Returns:
        X_train, X_val, Y_train, Y_val
    """    

    X = np.load("./data/X_train.npy")
    Y = np.load("./data/Y_train.npy")
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    return X_train, X_val, Y_train, Y_val

def scheduler(epoch, lr, patience=5):
    """
    Learning rate scheduler

    Args:
        patience (int): Patience of training loops
        epoch (int): Number of epochs
        lr (float): Learning rate

    Returns:
        _type_: _description_
    """    

    if epoch < patience:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def training():
    """
    Training process
    """ 
    # USE_GPU = 1  
    print(f'TensorFlow version: {tf.__version__}\n')

    # Get all GPU devices on this server
    gpu_devices = tf.config.list_physical_devices('GPU')

    # Print the name and the type of all GPU devices
    print('Available GPU Devices:')
    for gpu in gpu_devices:
        print(' ', gpu.name, gpu.device_type)
        
    # Set only the GPU specified as USE_GPU to be visible
    # tf.config.set_visible_devices(gpu_devices[USE_GPU], 'GPU')

    # Get all visible GPU  devices on this server
    visible_devices = tf.config.get_visible_devices('GPU')

    # Print the name and the type of all visible GPU devices
    print('\nVisible GPU Devices:')
    for gpu in visible_devices:
        print(' ', gpu.name, gpu.device_type)
        
    # Set the visible device(s) to not allocate all available memory at once,
    # but rather let the memory grow whenever needed
    for gpu in visible_devices:
        tf.config.experimental.set_memory_growth(gpu, True) 

    # Split the original data
    X_train, X_val, Y_train, Y_val = split_data()

    model = build_model()

    # Monitor the changes of validation accuracy to avoid overfitting
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2), 
                    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), batch_size=16, epochs=50, callbacks=callbacks)

    # Visualize the accuracy and loss during training process 
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1])
    plt.legend(loc='lower right')

    if not os.path.exists(os.getcwd()+"/figure"):
        os.makedirs(os.getcwd()+"/figure")
    plt.savefig("./figure/accuracy.png")
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig("./figure/loss.png")
    plt.show()

    # Save the model
    model.save('./model/UNet.h5')