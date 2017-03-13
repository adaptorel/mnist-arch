from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from keras.optimizers import RMSprop
from utils import build_training_data, callbacks


def build_partial_mlp_model(depth=12, wide_factor=22, dropout=0.15, init_mode='lecun_uniform', activation='relu'):
    deep_model = Sequential()
    deep_model.add(Flatten(input_shape=(28, 28)))
    deep_model.add(BatchNormalization())
    deep_model.add(Dense(wide_factor * depth, activation=activation, init=init_mode))
    deep_model.add(Dropout(dropout))
    for ind in range(1, depth, 1):
        if ind < 9:  # cut it a bit earlier
            deep_model.add(Dense(wide_factor * (depth - ind), activation=activation, init=init_mode))
            deep_model.add(Dropout(dropout))

    return deep_model


def train(lr=0.0075, nb_epoch=10, batch_size=256, verbose=1):
    X_train, y_train, X_test, y_test = build_training_data()
    model = build_partial_mlp_model()

    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=callbacks(), verbose=verbose)
    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train()
