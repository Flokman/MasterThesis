''' Trains a (pre trained) network with additional batch normalization layers
    for uncertainty estimation'''

import os
import datetime
import time
import random
import re
import h5py
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3
DATASET_NAME = 'CIFAR10'

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 150
MCBN_PREDICTIONS = 250
MINIBATCH_SIZE = 128
MCBN_BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = True
AUGMENTATION = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = True
ADD_BATCH_NORMALIZATION = True
ADD_BATCH_NORMALIZATION_INSIDE = True
TRAIN_ALL_LAYERS = False
ONLY_AFTER_SPECIFIC_LAYER = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.0001
ES_PATIENCE = 5

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + '/Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
        MCBN_PREDICTIONS = {}, Mini_batch_size = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        add_bn_inside = {}, train_all_layers = {}, weights_to_use = {},
        es_patience = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
            MCBN_PREDICTIONS, MINIBATCH_SIZE, test_img_idx,
            TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION,
            LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
            ADD_BATCH_NORMALIZATION_INSIDE, TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
            ES_PATIENCE, TRAIN_VAL_SPLIT))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return(x_train, y_train, x_test, y_test, test_img_idx)


def get_batch_normalization(input_tensor):
    ''' Returns a trainable batch normalization layer '''
    return BatchNormalization()(input_tensor, training=True)


def insert_intermediate_layer_in_keras(model, layer_id):
    ''' Insert a batch normalization layer before the layer with layer_id '''
    inter_layers = [l for l in model.layers]

    x = inter_layers[0].output
    for i in range(1, len(inter_layers)):
        if i == layer_id:
            x = get_batch_normalization(x)
        x = inter_layers[i](x)

    new_model = Model(inputs=inter_layers[0].input, outputs=x)
    return new_model


def add_batch_normalization(mcbn_model):
    ''' Adds batch normalizaiton layers either after all pool and dense layers
        or only after dense layers '''
    if ADD_BATCH_NORMALIZATION_INSIDE:
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in mcbn_model.layers])

        for layer_name in layer_dict:
            layer_dict = dict([(layer.name, layer) for layer in mcbn_model.layers])
            if ONLY_AFTER_SPECIFIC_LAYER:
                if re.search('.*_conv.*', layer_name):
                    print(layer_name)
                    layer_index = list(layer_dict).index(layer_name)
                    print(layer_index)

                    # Add a batch normalization (trainable) layer
                    mcbn_model = insert_intermediate_layer_in_keras(mcbn_model, layer_index + 1)

                    # mcbn_model.summary()
            else:
                print(layer_name)
                layer_index = list(layer_dict).index(layer_name)
                print(layer_index)

                # Add a batch normalization (trainable) layer
                mcbn_model = insert_intermediate_layer_in_keras(mcbn_model, layer_index + 1)



        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcbn_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        # x = get_batch_normalization(x)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_batch_normalization(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_batch_normalization(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    else:
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcbn_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_batch_normalization(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_batch_normalization(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Creating new model
    mcbn_model = Model(inputs=all_layers[0].input, outputs=x)
    return mcbn_model


def create_minibatch(x, y):
    ''' Returns a minibatch of the train data '''
    combined = list(zip(x, y)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    # print("Random seed minibatch for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
    minibatch = combined[:MINIBATCH_SIZE]

    (x_minibatch, y_minibatch) = zip(*minibatch)  
                            # *combined is used to separate all the tuples in the list combined,
                            # "x_minibatch" then contains all the shuffled images and
                            # "y_minibatch" contains all the shuffled labels.

    return(x_minibatch, y_minibatch)


def create_model(optimizer='adam'):
    # VGG16 since it does not include batch normalization of dropout by itself
    MCBN_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if ADD_BATCH_NORMALIZATION:
        MCBN_model = add_batch_normalization(MCBN_model)

    else:
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in MCBN_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

        # Creating new model.
        MCBN_model = Model(inputs=all_layers[0].input, outputs=x)

    if TRAIN_ALL_LAYERS or ADD_BATCH_NORMALIZATION_INSIDE:
        for layer in MCBN_model.layers:
            layer.trainable = True
            print(layer, layer.trainable)
    else:
        for layer in MCBN_model.layers[:-6]:
            layer.trainable = False
        for layer in MCBN_model.layers:
            print(layer, layer.trainable)

    MCBN_model.summary()

    MCBN_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return MCBN_model


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    MCBN_model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=32, verbose=0)

    print("Start fitting monte carlo batch_normalization model")

    X = x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))]
    X = X.astype('float32')
    Y = y_train[0:int(TRAIN_VAL_SPLIT*len(x_train))]
    X /= 255

    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=MCBN_model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main()
