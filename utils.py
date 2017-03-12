from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau

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