from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import RMSprop

from utils import build_training_data, callbacks


def build_partial_lstm_model(neurons=32, dropout=0.45):
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(neurons, dropout_W=dropout), merge_mode='concat', input_shape=(28, 28)))
    lstm_model.add(Dropout(dropout))
    return lstm_model


def train(lr=0.0075, nb_epoch=10, batch_size=512, verbose=1):
    X_train, y_train, X_test, y_test = build_training_data()

    lstm_model = build_partial_lstm_model()
    lstm_model.add(Dense(10, activation='softmax'))
    lstm_model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    lstm_model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_test, y_test),
                   callbacks=callbacks(), verbose=verbose)
    score = lstm_model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train()
