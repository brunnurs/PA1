import numpy as np

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from active_learning.metrics import Metrics

from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from persistance.pickle_service import PickleService

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras

tf.set_random_seed(42)
verbose = 1


def explore_fcNN_performance(data, gold_standard):
    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)
    # x, y = downsample_to_even_classes(data)

    # do_cross_validation(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    print('train-set shape: {}, {}'.format(np.shape(x_train), np.shape(y_train)))
    print('test-set shape: {}, {}'.format(np.shape(x_test), np.shape(y_test)))

    # x_train, y_train = SMOTE_oversampling(x_train, y_train)

    model = create_keras_model()
    train_evaluate_model(model, x_test, x_train, y_test, y_train)


def do_cross_validation(x, y):
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    counter = 1
    for train_index, test_index in skf.split(x, y):
        print("====================================================================")
        print("Running fold {} of {} folds".format(counter, n_folds))

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = create_keras_model()

        train_evaluate_model(model, x_test, x_train, y_test, y_train)

        counter += 1


def train_evaluate_model(model, x_test, x_train, y_test, y_train):
    # model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=verbose, class_weight={0: 1.0, 1: 19.0})
    model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=verbose)

    y_pred = model.predict_classes(x_test, batch_size=128)

    Metrics.print_classification_report_raw(y_pred, y_test)

    probas_pred = model.predict(x_test, batch_size=128, verbose=verbose)
    Metrics.plot_precision_recall_curve(y_test, probas_pred)


def create_keras_model():
    model = Sequential()

    model.add(Dense(500, batch_input_shape=(None, 4)))
    # model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(500))
    # model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(50))
    # model.add(keras.layers.normalization.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    # WeightedCategoricalCrossEntropy({0: 1.0, 1: 19.0})

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

    if verbose == 1:
        model.summary()

    return model


if __name__ == "__main__":
    print('====== fcNN (with Tensorflow and Keras) performance ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_fcNN_performance(ps, gs)
