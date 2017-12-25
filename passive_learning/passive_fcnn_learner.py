import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from active_learning.metrics import Metrics
from active_learning.utils import transform_to_labeled_feature_vector
from data_analytics.manual_analytics import save_information_about_predictions
from passive_learning.passive_learner_utils import label_data
from passive_learning.sampling import random_undersampling, SMOTE_oversampling
from persistance.pickle_service import PickleService

import pickle

tf.set_random_seed(42)
verbose = 1


def explore_fcNN_performance(data, gold_standard):
    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)
    # x, y = random_undersampling(x, y)
    indices = range(len(x))

    # activate me if working with word-vector similarities!
    # x = MaxAbsScaler().fit_transform(x)

    # do_cross_validation(x, y)

    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, indices, test_size=0.25,
                                                                                     random_state=42, stratify=y)

    print('train-set shape: {}, {}'.format(np.shape(x_train), np.shape(y_train)))
    print('test-set shape: {}, {}'.format(np.shape(x_test), np.shape(y_test)))

    # x_train, y_train = SMOTE_oversampling(x_train, y_train)

    model = create_keras_model(np.shape(x)[1])
    train_evaluate_model(model, x_test, x_train, y_test, y_train, do_plot=True)

    # save_information_about_predictions(model, x_test, y_test, indices_test, data)


def do_cross_validation(x, y):
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    counter = 1
    for train_index, test_index in skf.split(x, y):
        print("====================================================================")
        print("Running fold {} of {} folds".format(counter, n_folds))

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_keras_model(len(x))

        train_evaluate_model(model, x_test, x_train, y_test, y_train, do_plot=False)

        counter += 1


def train_evaluate_model(model, x_test, x_train, y_test, y_train, do_plot=True):
    # history = model.fit(x_train, y_train
    #                     , epochs=13
    #                     , batch_size=128
    #                     , verbose=verbose
    #                     , validation_split=0.2)

    history = model.fit(x_train, y_train
                        , epochs=20
                        , batch_size=128
                        , verbose=verbose
                        , validation_split=0.2)

    y_pred = model.predict_classes(x_test, batch_size=128)

    Metrics.print_classification_report_raw(y_pred, y_test)

    if do_plot:
        probas_pred = model.predict(x_test, batch_size=128, verbose=verbose)
        Metrics.plot_precision_recall_curve(y_test, probas_pred)

        import matplotlib.pyplot as plt

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train acc', 'test acc'], loc='lower right')

        plt.show()


def create_keras_model(input_vector_length):
    print('Input vector length is {}'.format(input_vector_length))

    model = Sequential()

    model.add(Dense(500, batch_input_shape=(None, input_vector_length)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(500))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(500))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    if verbose == 1:
        model.summary()

    return model


if __name__ == "__main__":
    print('====== fcNN (with Tensorflow and Keras) performance ======')

    pickle_service = PickleService()
    ps = pickle_service.load_pre_processed_data('./data/intermediate_data')
    gs = pickle_service.load_gold_standard_data('./data/intermediate_data')

    explore_fcNN_performance(ps, gs)
