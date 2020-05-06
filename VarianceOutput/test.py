''' Trains a (pre trained) network with additional dropout layers for uncertainty estimation'''

import os
import datetime
import time
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras import backend as K

from uncertainty_output import Uncertainty_output

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3
DATASET_NAME = os.path.sep + 'CIFAR10'

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS_1 = 50
ES_PATIENCE_1 = 10
EPOCHS_2 = 50
ES_PATIENCE_2 = 15

TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9

TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001

SCATTER = True

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + os.path.sep + 'Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES*2)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES*2)

    return(x_train, y_train, x_test, y_test)


def scatterplot(accuracies, uncertainties, methodname, own_or_new):
    # os.chdir(HOME_DIR + os.path.sep + methodname)

    plt.scatter(accuracies, uncertainties)
    plt.xlabel('probability')
    plt.ylabel('uncertainty')
    plt.title('Scatterplot for {} on {}'.format(methodname, own_or_new))
    plt.savefig('{}_scatter_{}.png'.format(methodname, own_or_new))
    plt.clf()

    # os.chdir(HOME_DIR)


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test = prepare_data()

    # VGG16 since it does not include batch normalization of dropout by itself
    variance_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
                       classes=NUM_CLASSES)

    # Stacking a new simple convolutional network on top of vgg16 (as demo, last layer is replaced later again)
    # Classification block
    x = Flatten(name='flatten')(variance_model.layers[-1].output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation = 'softmax')(x)

    variance_model = Model(variance_model.input, [x])

    variance_model.summary()

    # Replacing last layer existing model by error_output layer
    variance_model = Uncertainty_output(NUM_CLASSES).create_uncertainty_model(variance_model)

    variance_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)


    # Compile with custom loss function categorica_cross to first only train the classification outputs
    variance_model.compile(
        optimizer=adam,
        loss=Uncertainty_output(NUM_CLASSES).categorical_cross,
        metrics=['acc']
    )

    print("Start fitting")
    
    early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_1)


    datagen = ImageDataGenerator(rescale=1./255, dtype='ndarray')
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)

    # First training round, because of loss function only the classification is trained
    variance_model.fit(train_generator,
                epochs=EPOCHS_1,
                verbose=2,
                validation_data=val_generator,
                callbacks=[early_stopping_1])


    # Compile with custom loss function categorical_error to first only train the classification outputs
    variance_model.compile(
        optimizer=adam,
        loss=Uncertainty_output(NUM_CLASSES).categorical_error,
        metrics=['acc']
    )

    early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_2)


    # Second training round, now all outputs are trained
    # (Potentially make a division on the trainset to train both rounds on unique data)
    variance_model.fit(train_generator,
                       epochs=EPOCHS_2,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[early_stopping_2])


    variance_predictions = variance_model.predict(x_test)
    variance_predictions = Uncertainty_output(NUM_CLASSES).convert_output_to_uncertainty(variance_predictions, y_test)
    
    Uncertainty_output(NUM_CLASSES).results_if_label(variance_predictions, y_test)

    if SCATTER:
        # Info of all classes
        all_probabilities = []
        all_uncertainties = []

        true_labels = [np.argmax(i) for i in y_test]

        for ind, pred in enumerate(variance_predictions):
            true_label = true_labels[ind]
            predictions = pred[:NUM_CLASSES]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = pred[NUM_CLASSES:]

            for class_ind, (prob, unc) in enumerate(zip(predictions, uncertainties)):
                all_probabilities.append(prob)
                all_uncertainties.append(unc)

        scatterplot(all_probabilities, all_uncertainties, 'CIFAR10', 'DEMO')

if __name__ == "__main__":
    main()
