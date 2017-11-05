if __name__ == "__main__":
    '''Train a simple deep NN on the MNIST dataset
    using custom loss function with costs'''

    import numpy as np

    np.random.seed(1337)  # for reproducibility

    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils
    import keras.backend as K
    from itertools import product


    # Custom loss function with costs

    def w_categorical_crossentropy(weights):
        def loss(y_true, y_pred):
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1, keepdims=True)
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
            return K.categorical_crossentropy(y_pred, y_true) * final_mask

        return loss


    batch_size = 128
    epochs = 2  # small for demo purposes

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Fit neural network with 10 classes (no problem using custom loss function)

    nb_classes = 10
    wcc = w_categorical_crossentropy(np.ones((nb_classes, nb_classes)))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss=wcc, optimizer=rms, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])