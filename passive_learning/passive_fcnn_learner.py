import tensorflow as tf
from keras.metrics import binary_accuracy
from sklearn.metrics import precision_score, recall_score, f1_score

from active_learning.metrics import Metrics
from passive_learning.sampling import downsample_to_even_classes

tf.set_random_seed(42)

from sklearn.model_selection import train_test_split

from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from persistance.pickle_service import PickleService

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras

def explore_fcNN_performance(data, gold_standard):
    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)
    x, y = downsample_to_even_classes(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    print('train-set shape: {}, {}'.format(np.shape(x_train), np.shape(y_train)))
    print('test-set shape: {}, {}'.format(np.shape(x_test), np.shape(y_test)))

    model = create_keras_model()

    history = model.fit(x_train, y_train, epochs=100, batch_size=128)

    y_pred = model.predict_classes(x_test, batch_size=128)

    Metrics.print_classification_report_raw(y_pred, y_test)

    # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    # print(loss_and_metrics)

    # plt.plot(loss_and_metrics['acc'])
    # plt.plot(loss_and_metrics['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train acc', 'test acc'], loc='lower right')
    # plt.show()


def create_keras_model():
    model = Sequential()

    model.add(Dense(500, batch_input_shape=(None, 2)))
    # model.add(keras.layers.normalization.BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(50))
    # model.add(keras.layers.normalization.BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

    model.summary()

    return model


if __name__ == "__main__":
    print('====== fcNN (with Tensorflow and Keras) performance ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_fcNN_performance(ps, gs)
