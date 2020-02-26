''' Trains a (pre trained) network with additional dropout layers for uncertainty estimation'''

#https://stackoverflow.com/questions/49646304/keras-optimizing-two-outputs-with-a-custom-loss
# https://stackoverflow.com/questions/46663013/what-is-y-true-and-y-pred-when-creating-a-custom-metric-in-keras

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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils, backend

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3
DATASET_NAME = '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'

BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 500
N_ENSEMBLE_MEMBERS = 40
AMOUNT_OF_PREDICTIONS = 50
TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = True
AUGMENTATION = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = True
TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = None
LEARN_RATE = 0.00001
ES_PATIENCE = 30

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + '/Datasets' + DATASET_NAME

def shuffle_data(x_to_shuff, y_to_shuff):
    ''' Shuffle the data randomly '''
    # use zip() to bind the images and label together
    combined = list(zip(x_to_shuff, y_to_shuff))
    random_seed = random.randint(0, 1000)
    print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)

    (x_shuffled, y_shuffled) = zip(*combined)
        # *combined is used to separate all the tuples in the list combined,
        # "x_shuffled" then contains all the shuffled images and
        # "y_shuffled" contains all the shuffled labels.
    return (x_shuffled, y_shuffled)


def data_augmentation(x_aug, y_aug, label_count):
    ''' Augment the data according to the settings of imagedatagenerator '''
    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    if LABEL_NORMALIZER:
        start_aug = time.time()
        # Also perform some augmentation on the class with most examples
        goal_amount = int(max(label_count) * 1.2)
        print("goal amount", goal_amount)

        # Sort by class
        sorted_tup = []
        for num in range(0, NUM_CLASSES):
            sorted_tup.append([])
        for idx, label in enumerate(y_aug):
            sorted_tup[label].append((x_aug[idx], y_aug[idx]))

        for label, label_amount in enumerate(label_count):
            # Divided by 5 since 5 augmentations will be performed at a time
            amount_to_augment = int((goal_amount - label_amount)/5)
            print("amount to aug", amount_to_augment*NUM_CLASSES)
            print("class", label)
            tups_to_augment = random.choices(sorted_tup[label], k=amount_to_augment)

            to_augment_x = [(tups_to_augment[0])[0]]
            to_augment_y = np.empty(amount_to_augment)
            to_augment_y.fill(label)

            x_org_len = len(x_aug)

            for i in range(1, amount_to_augment):
                to_augment_x = np.append(to_augment_x, [(tups_to_augment[i])[0]], axis=0)

            for i in range(0, amount_to_augment): #amount to augment
                i = 0
                # for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1,
                #             save_to_dir ='preview/' + str(label),
                #             save_prefix = str(label) + '_image', save_format ='png'):
                for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1):
                    x_aug = np.append(x_aug[:], batch[0], axis=0)
                    y_aug = np.append(y_aug[:], batch[1], axis=0)
                    print("{}/{}".format((len(x_aug) - x_org_len), amount_to_augment*NUM_CLASSES))
                    i += 1
                    if i > 5:
                        break

        norm_done = time.time()
        print("Augmenting normalization finished after {0} seconds".format(norm_done - start_aug))
        label_count = [0] * NUM_CLASSES
        for lab in y_aug:
            label_count[int(lab)] += 1
        print('label count after norm:', label_count)

    if SAVE_AUGMENTATION_TO_HDF5:
        start_sav = time.time()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # file path for the created .hdf5 file
        hdf5_path = dir_path + '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        print(hdf5_path)

        # open a hdf5 file and create earrays
        f = h5py.File(hdf5_path, mode='w')

        f.create_dataset("x_aug", data=x_aug, dtype='uint8')
        f.create_dataset("y_aug", data=y_aug, dtype='uint8')

        f.close()
        save_done = time.time()
        print("Saving finished after {0} seconds".format(save_done - start_sav))
    return (x_aug, y_aug)


def load_data(path, to_shuffle):
    '''' Load a dataset from a hdf5 file '''
    with h5py.File(path, "r") as f:
        (x_load, y_load) = np.array(f['x']), np.array(f['y'])
    label_count = [0] * NUM_CLASSES
    for lab in y_load:
        label_count[lab] += 1

    if to_shuffle:
        (x_load, y_load) = shuffle_data(x_load, y_load)

    if AUGMENTATION:
        (x_load, y_load) = data_augmentation(x_load, y_load, label_count)
        print("augmentation done")
        label_count = [0] * NUM_CLASSES
        for lab in y_load:
            label_count[lab] += 1


    # Divide the data into a train and test set
    x_train = x_load[0:int(TRAIN_TEST_SPLIT*len(x_load))]
    y_train = y_load[0:int(TRAIN_TEST_SPLIT*len(y_load))]

    x_test = x_load[int(TRAIN_TEST_SPLIT*len(x_load)):]
    y_test = y_load[int(TRAIN_TEST_SPLIT*len(y_load)):]

    return (x_train, y_train), (x_test, y_test), label_count


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test), label_count = load_data(DATA_PATH, TO_SHUFFLE)

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
          test_img_idx = {}, train_test_split = {}, to_shuffle = {},
          augmentation = {}, label_count = {}, label_normalizer = {},
          save_augmentation_to_hdf5 = {}, learn rate = {}, train_all_layers = {},
          weights_to_use = {}, es_patience = {}, train_val_split = {},
          N_ENSEMBLE_MEMBERS = {}""".format(
              DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
              test_img_idx, TRAIN_TEST_SPLIT, TO_SHUFFLE,
              AUGMENTATION, label_count, LABEL_NORMALIZER,
              SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE, TRAIN_ALL_LAYERS,
              WEIGHTS_TO_USE, ES_PATIENCE, TRAIN_VAL_SPLIT,
              N_ENSEMBLE_MEMBERS))

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


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    # VGG16 since it does not include batch normalization of dropout by itself
    variance_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
                       classes=NUM_CLASSES)

    # Stacking a new simple convolutional network on top of vgg16
    all_layers = [l for l in variance_model.layers]
    x = all_layers[0].output
    for i in range(1, len(all_layers)):
        x = all_layers[i](x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    last_layer = Dense(4096, activation='relu', name='fc2')(x)
    classifier = Dense(NUM_CLASSES, activation='softmax', name='classification')(last_layer)

    # Regression layer
    regression = Dense(NUM_CLASSES, activation='linear', name='variance')(last_layer)

    # Creating new model
    variance_model = Model(inputs=all_layers[0].input, outputs=[classifier, regression])







    variance_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)
    variance_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Start fitting")

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)


    x_axis, y_axis = list(range(len(p_0.mean(axis=0)))), p_0.mean(axis=0)
    plt.plot(x_axis, y_axis)
    plt.savefig('prob_var_' + str(test_img_idx) + '.png')
    plt.clf()

    fig = plt.subplots(5, 1, figsize=(12, 6))[0]

    for i, ax in enumerate(fig.get_axes()):
        ax.hist(p_0[:, i], bins=100, range=(0, 1))
        ax.set_title(f"class {i}")
        ax.label_outer()

    fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)


if __name__ == "__main__":
    main()
