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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score


# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 75, 75, 3
DATASET_NAME = 'CIFAR10'

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 500
N_ENSEMBLE_MEMBERS = 3
AMOUNT_OF_PREDICTIONS = 50
TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.875
SAVE_AUGMENTATION_TO_HDF5 = True

TO_SHUFFLE = False
AUGMENTATION = False
LABEL_NORMALIZER = False
TRAIN_ALL_LAYERS = True

WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001
ES_PATIENCE = 20
RANDOMSEED = None
MIN_DELTA = 0.005
EARLY_MONITOR = 'val_accuracy'
MC_MONITOR = 'val_loss'
RESULTFOLDER = 'CIFAR10'

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + '/Datasets' + DATASET_NAME

def shuffle_data(x_to_shuff, y_to_shuff):
    ''' Shuffle the data randomly '''
    # use zip() to bind the images and label together
    combined = list(zip(x_to_shuff, y_to_shuff))
    if RANDOMSEED != None:
        random_seed = RANDOMSEED
        print("Copied seed for replication: {}".format(random_seed))
    else:
        random_seed = random.randint(0, 1000)
        print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)

    (x_shuffled, y_shuffled) = zip(*combined)
        # *combined is used to separate all the tuples in the list combined,
        # "x_shuffled" then contains all the shuffled images and
        # "y_shuffled" contains all the shuffled labels.
    return (x_shuffled, y_shuffled)


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
          test_img_idx = {}, train_test_split = {}, to_shuffle = {},
          augmentation = {}, label_normalizer = {},
          save_augmentation_to_hdf5 = {}, learn rate = {}, train_all_layers = {},
          weights_to_use = {}, es_patience = {}, train_val_split = {},
          N_ENSEMBLE_MEMBERS = {}, MIN_DELTA = {}, Early_monitor = {}""".format(
              DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
              test_img_idx, TRAIN_TEST_SPLIT, TO_SHUFFLE,
              AUGMENTATION, LABEL_NORMALIZER,
              SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE, TRAIN_ALL_LAYERS,
              WEIGHTS_TO_USE, ES_PATIENCE, TRAIN_VAL_SPLIT,
              N_ENSEMBLE_MEMBERS, MIN_DELTA, EARLY_MONITOR))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)

    x_train_re = np.empty((len(x_train), IMG_HEIGHT, IMG_HEIGHT, 3))
    x_test_re = np.empty((len(x_test), IMG_HEIGHT, IMG_HEIGHT, 3))

    for ind, img in enumerate(x_train):
        x_train_re[ind, ...] = pad(img, IMG_HEIGHT, IMG_HEIGHT)

    for ind, img in enumerate(x_test):
        x_test_re[ind, ...] = pad(img, IMG_HEIGHT, IMG_HEIGHT)


    print('x_train_re shape:', x_train_re.shape)
    print(x_train_re.shape[0], 'train samples')
    print(x_test_re.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return(x_train_re, y_train, x_test_re, y_test, test_img_idx)


def fit_model(x_train_splits, y_train_splits, x_val, y_val, ensemble_model, log_dir, i, architecture):
    ensemble_model.load_weights('initial_weights.h5')

    print("Length split part: ", len(x_train_splits[i]))

    label_count = [0] * NUM_CLASSES
    for lab in y_train_splits[i]:
        label_count[np.argmax(lab)] += 1
    print("Labels in this part of split: ", label_count)    

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(x_train_splits[i],
                                   y_train_splits[i],
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_val,
                                 y_val,
                                 batch_size=BATCH_SIZE)

    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor=MC_MONITOR, mode='auto', save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=EARLY_MONITOR, min_delta = MIN_DELTA,
                                                    mode='auto', verbose=1, patience=ES_PATIENCE)

    ensemble_model.fit(train_generator,
                       epochs=EPOCHS,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback, early_stopping, mc])

    ensemble_model = load_model('best_model.h5')
    os.remove('best_model.h5')

    # Save JSON config to disk
    json_config = ensemble_model.to_json()
    with open("ensemble_model_config_{}_{}.json".format(architecture, i), 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    ensemble_model.save_weights("ensemble_weights_{}_{}.h5".format(architecture, i))

    # To save memory, clear memory and return something else
    K.clear_session()

    return 1


def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))


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
    



    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    archi_list = [Xception, VGG16, VGG19, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201]
    archi_name = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201']
    for ind, architecture in enumerate(archi_list):
        print(archi_name[ind])

        # use all architectures defined in archi_list
        ensemble_model = architecture(weights=WEIGHTS_TO_USE, include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


        # Stacking a new simple convolutional network on top of vgg16
        x = ensemble_model.output

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

        # Creating new model
        ensemble_model = Model(inputs=ensemble_model.input, outputs=x)

        if TRAIN_ALL_LAYERS:
            for layer in ensemble_model.layers:
                layer.trainable = True
        else:
            for layer in ensemble_model.layers[:-6]:
                layer.trainable = False
            for layer in ensemble_model.layers:
                print(layer, layer.trainable)

        ensemble_model.summary()

        adam = optimizers.Adam(lr=LEARN_RATE)
        # sgd = optimizers.SGD(lr=LEARN_RATE)

        ensemble_model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Start fitting ensemble models")

        # Save initial weights
        ensemble_model.save_weights('initial_weights.h5')

        # Split train dataset so every enemble can train on an unique part of the data for maximum variance between the models
        x_train_splits = np.array_split(x_train, N_ENSEMBLE_MEMBERS)
        y_train_splits = np.array_split(y_train, N_ENSEMBLE_MEMBERS)

        ensemble = [fit_model(x_train_splits, y_train_splits, x_val, y_val, ensemble_model, log_dir, i, archi_name[ind]) for i in range(N_ENSEMBLE_MEMBERS)]

        os.remove('initial_weights.h5')


if __name__ == "__main__":
    main()
