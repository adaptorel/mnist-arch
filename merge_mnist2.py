from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Merge
from keras.optimizers import RMSprop
from utils import build_training_data, callbacks

from utils import build_training_data
from mlp_mnist import build_partial_mlp_model
from lstm_mnist import build_partial_lstm_model


def train(lr=0.0075, nb_epoch=10, batch_size=512, verbose=1):
    X_train, y_train, X_test, y_test = build_training_data()
    model = Sequential()

    lstm = build_partial_lstm_model()
    mlp = build_partial_mlp_model()

    # To SUM you'll have to match the outputs of the partial networks to be the same size, aka 64 as it is now
    # model.add(Merge([mlp, lstm], mode='sum'))
    # Concat will work with different sizes
    model.add(Merge([mlp, lstm], mode='concat'))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([X_train, X_train], y_train, nb_epoch=nb_epoch, batch_size=batch_size,
              validation_data=([X_test, X_test], y_test), callbacks=callbacks(), verbose=verbose)
    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train()
