''' Trains a (pre trained) network with additional dropout layers for uncertainty estimation'''

#https://stackoverflow.com/questions/49646304/keras-optimizing-two-outputs-with-a-custom-loss
# https://stackoverflow.com/questions/46663013/what-is-y-true-and-y-pred-when-creating-a-custom-metric-in-keras
# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras import backend as K

from uncertainty_output import Uncertainty_output

# from customLoss import CategoricalVariance

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3
DATASET_NAME = os.path.sep + 'CIFAR10'

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS_1 = 150
ES_PATIENCE_1 = 10
EPOCHS_2 = 30
ES_PATIENCE_2 = 10
EPOCHS_3 = 5
ES_PATIENCE_3 = 2

TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.875

TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001


# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + os.path.sep + 'Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs_1 = {},
        epochts_2 = {}, test_img_idx = {},
        train_test_split = {}
        learn rate = {},
        train_all_layers = {}, weights_to_use = {},
        es_patience_1 = {}, es_patience_2 = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS_1,
            EPOCHS_2, test_img_idx,
            TRAIN_TEST_SPLIT,
            LEARN_RATE,
            TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
            ES_PATIENCE_1, ES_PATIENCE_2, TRAIN_VAL_SPLIT))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES*2)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES*2)

    return(x_train, y_train, x_test, y_test, test_img_idx)


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    x_test, x_val = np.split(x_test, [int(TRAIN_VAL_SPLIT*len(x_test))])
    y_test, y_val = np.split(y_test, [int(TRAIN_VAL_SPLIT*len(y_test))])

    label_count = [0] * NUM_CLASSES
    for lab in y_train:
        label_count[np.argmax(lab)] += 1
    print("Total labels in train set: ", label_count)   

    label_count = [0] * NUM_CLASSES
    for lab in y_val:
        label_count[np.argmax(lab)] += 1
    print("Labels in validation set: ", label_count)   
    
    label_count = [0] * NUM_CLASSES
    for lab in y_test:
        label_count[np.argmax(lab)] += 1
    print("Labels in test set: ", label_count) 

    # VGG16 since it does not include batch normalization of dropout by itself
    Error_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
                       classes=NUM_CLASSES)

    # Stacking a new simple convolutional network on top of vgg16
    all_layers = [l for l in Error_model.layers]
    x = all_layers[0].output
    for i in range(1, len(all_layers)):
        x = all_layers[i](x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    last_layer_1 = Dense(4096, activation='relu', name='fc2_1')(x)
    last_layer_2 = Dense(4096, activation='relu', name='fc2_2')(x)
    classification = Dense(NUM_CLASSES, activation='softmax')(last_layer_1)
    Error = Dense(NUM_CLASSES, activation='softmax')(last_layer_2)

    out = concatenate([classification, Error])

    # Creating new model
    Error_model = Model(inputs=all_layers[0].input, outputs=out)

    Error_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)

    Error_model.compile(
        optimizer=adam,
        loss=Uncertainty_output(NUM_CLASSES).error,
        metrics=['acc']
    )

    print("Start fitting")
    
    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), "CIFAR10" + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    mc_1 = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='loss', mode='min', save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_1)


    datagen = ImageDataGenerator(rescale=1./255, dtype='ndarray')
    train_generator = datagen.flow(x_train,
                                   y_train,
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_val,
                                 y_val,
                                 batch_size=BATCH_SIZE)



    Error_model.fit(train_generator,
                epochs=EPOCHS_1,
                verbose=2,
                validation_data=val_generator,
                callbacks=[tensorboard_callback, early_stopping_1, mc_1])

    Error_model = load_model('best_model.h5', compile = False)
    os.remove('best_model.h5')

    Error_model.compile(
        optimizer=adam,
        loss=Uncertainty_output(NUM_CLASSES).categorical_cross,
        metrics=['acc']
    )

    mc_2 = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_2)

    Error_model.fit(train_generator,
                       epochs=EPOCHS_2,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback, early_stopping_2, mc_2])

    Error_model = load_model('best_model.h5', compile = False)
    os.remove('best_model.h5')

    Error_model.compile(
        optimizer=adam,
        loss=Uncertainty_output(NUM_CLASSES).categorical_error,
        metrics=['acc']
    )

    mc_3 = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='loss', mode='min', save_best_only=True) 
    early_stopping_3 = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_2)

    Error_model.fit(train_generator,
                       epochs=EPOCHS_2,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback, early_stopping_2, mc_3])

    Error_model = load_model('best_model.h5', compile = False)
    os.remove('best_model.h5')

    # Save JSON config to disk
    json_config = Error_model.to_json()
    with open('Error_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    Error_model.save_weights('Error_weights.h5')

    Error_predictions = Error_model.predict(x_test)
    Uncertainty_output(NUM_CLASSES).results_if_label(Error_predictions, y_test, scatter=True, name='CIFAR10_No_Conv')

    Error_predictions = Uncertainty_output(NUM_CLASSES).convert_output_to_uncertainty(Error_predictions)
    Uncertainty_output(NUM_CLASSES).results_if_label(Error_predictions, y_test, scatter=True, name='CIFAR10')


    single_acc = accuracy_score(y_test.argmax(axis=1), Error_predictions[:, :NUM_CLASSES].argmax(axis=1))
    print(single_acc)
    dir_path_head_tail = os.path.split(os.path.dirname(os.getcwd()))
    if WEIGHTS_TO_USE != None:
        new_path = dir_path_head_tail[0] + os.path.sep + 'CIFAR10' + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + WEIGHTS_TO_USE + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    else:
        new_path = dir_path_head_tail[0] + os.path.sep + 'CIFAR10' + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    os.rename(fig_dir, new_path)


if __name__ == "__main__":
    main()
