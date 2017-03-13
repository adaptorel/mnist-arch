from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau
import re
import os

# No need to plot if you hate matplotlib
try:
    import matplotlib
    import sys
    matplotlib.use('TkAgg' if 'darwin' in sys.platform else 'Agg')
    import matplotlib.pyplot as plt
except Exception as ex:
    print("You'll have to install matplotlib if you want to save some pretty graphs")


def build_training_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test

def callbacks():
    reduce_lr = ReduceLROnPlateau(factor=0.8, patience=3, min_lr=0.00001, verbose=1)
    return [reduce_lr]

def plot_results(history, accuracy, title):
    graphs_dir = '_graphs'
    if not os.path.isdir(graphs_dir):
        os.mkdir(graphs_dir)
    try:
        plt.figure(figsize=(12,10))
        plt.subplot(311)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(['Accuracy', 'Validation accuracy'], loc='center right')
        plt.ylabel('Accuracy')
        plt.grid()

        plt.title("{} Test Accuracy: {:.2f}%".format(title.upper(), accuracy*100))

        plt.subplot(312)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Loss', 'Validation loss'], loc='center right')
        plt.ylabel('Loss')
        plt.grid()

        plt.subplot(313)
        plt.plot(history.history['lr'])
        plt.legend(['Learning Rate'], loc='center right')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid()

        plt.savefig('{}/{}_mnist.png'.format(graphs_dir, re.sub('\W', '_', title)).lower())
    except:
        print("You'll have to install matplotlib if you want to save some pretty graphs")