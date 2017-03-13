from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Merge
from keras.optimizers import RMSprop
from utils import build_training_data, callbacks, plot_results

from mlp_mnist import build_partial_mlp_model
from lstm_mnist import build_partial_lstm_model


# The merge arch will sometimes not converge, it's Twin Peaks, usually changing a parameter, any of them, will nudge it
# to converge. Another way to nudge it is to quickly uninstall/install Tensorflow. Yeah, I know, don't even ask.
def train(lr=0.0075, nb_epoch=10, batch_size=256, verbose=1):
    X_train, y_train, X_test, y_test = build_training_data()
    model = Sequential()

    lstm = build_partial_lstm_model()
    mlp = build_partial_mlp_model()

    # To SUM you'll have to match the outputs of the partial networks to be the same size, aka 64 as it is now
    # Also to be able to SUM/MEAN etc. we need to oversize the LSTM a bit to match the output of the MLP so if
    # the LSTM overfits a bit at the end, now you know why
    # model.add(Merge([mlp, lstm], mode='sum'))
    # Concat will work with different sizes
    model.add(Merge([mlp, lstm], mode='concat'))

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([X_train, X_train], y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                        validation_data=([X_test, X_test], y_test), callbacks=callbacks(), verbose=verbose)
    score = model.evaluate([X_test, X_test], y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    plot_results(history, score[1], 'MLP w/ LSTM')


if __name__ == '__main__':
    train()
