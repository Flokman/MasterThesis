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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score

from multiprocessing import Process, Queue


# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3
DATASET_NAME = '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'

BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 500
N_ENSEMBLE_MEMBERS = 3
AMOUNT_OF_PREDICTIONS = 50
TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.875
TEST_VAL_SPLIT = 0.66
SAVE_AUGMENTATION_TO_HDF5 = False

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
RESULTFOLDER = 'MES'
MC_MONITOR = 'val_loss'

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


def data_augmentation(x_train, y_train, x_test, y_test):
    ''' Augment the data according to the settings of imagedatagenerator '''
    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    if LABEL_NORMALIZER:
        print(x_train.shape, y_train.shape)
        # x_train, x_val = np.split(x_train, [int(TRAIN_VAL_SPLIT*len(x_train))])
        # y_train, y_val = np.split(y_train, [int(TRAIN_VAL_SPLIT*len(y_train))])
        # print(x_train.shape, y_train.shape)
        # print(x_val.shape, y_val.shape)
        start_aug = time.time()


        # Sort by class
        sorted_imgs = []
        for num in range(0, NUM_CLASSES):
            sorted_imgs.append([])
        for idx, label in enumerate(y_train):
            sorted_imgs[label].append(x_train[idx])

        label_count = [0] * NUM_CLASSES
        for i in range(0, NUM_CLASSES):
            label_count[i] = len(sorted_imgs[i])
        print(label_count)

        # Also perform some augmentation on the class with most examples
        goal_amount = int(max(label_count) * 1.2)
        print("goal amount", goal_amount)
        
        for label, label_amount in enumerate(label_count):
            print("label {}, label_amount {}".format(label, label_amount))
            # Divided by 5 since 5 augmentations will be performed at a time
            amount_to_augment = int((goal_amount - label_amount)/5)
            print("amount to aug", amount_to_augment*NUM_CLASSES)
            print("class", label)
            imgs_to_augment = random.choices(sorted_imgs[label], k=amount_to_augment)

            to_augment_x = [imgs_to_augment[0]]
            to_augment_y = np.empty(amount_to_augment)
            to_augment_y.fill(label)

            x_org_len = len(x_train)

            for i in range(1, amount_to_augment):
                to_augment_x = np.append(to_augment_x, [(imgs_to_augment[i])], axis=0)

            print(to_augment_x.shape)

            for i in range(0, amount_to_augment): #amount to augment
                i = 1
                # for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1,
                #             save_to_dir ='preview/' + str(label),
                #             save_prefix = str(label) + '_image', save_format ='png'):
                for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1):
                    x_train = np.append(x_train[:], batch[0], axis=0)
                    y_train = np.append(y_train[:], batch[1], axis=0)
                    print("{}/{}".format((len(x_train) - x_org_len), amount_to_augment*NUM_CLASSES))
                    i += 1
                    if i > 5:
                        break

        norm_done = time.time()
        print("Augmenting normalization finished after {0} seconds".format(norm_done - start_aug))
        label_count = [0] * NUM_CLASSES
        for lab in y_train:
            label_count[int(lab)] += 1
        print('label count after norm:', label_count)
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        # x_train = np.vstack((x_train, x_val))
        # y_train = np.append(y_train, y_val)
        # print(x_train.shape, y_train.shape)
        

    if SAVE_AUGMENTATION_TO_HDF5:
        start_sav = time.time()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # file path for the created .hdf5 file
        hdf5_path = dir_path + '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        print(hdf5_path)

        # open a hdf5 file and create earrays
        f = h5py.File(hdf5_path, mode='w')

        f.create_dataset("x_train", data=x_train, dtype='uint8')
        f.create_dataset("y_train", data=y_train, dtype='uint8')
        f.create_dataset("x_test", data=x_test, dtype='uint8')
        f.create_dataset("y_test", data=y_test, dtype='uint8')

        f.close()
        save_done = time.time()
        print("Saving finished after {0} seconds".format(save_done - start_sav))
    return (x_train, y_train)


def load_data(path, to_shuffle):
    '''' Load a dataset from a hdf5 file '''
    with h5py.File(path, "r") as f:
        x_train, y_train, x_test, y_test = np.array(f['x_train']), np.array(f['y_train']), np.array(f['x_test']), np.array(f['y_test'])
    label_count = [0] * NUM_CLASSES
    for lab in y_train:
        label_count[lab] += 1

    print("loaded data")
    if to_shuffle:
        (x_load, y_load) = shuffle_data(x_load, y_load)

    # Divide the test data (not augmented) into a validation and test set
    x_test, x_val = np.split(x_test, [int(TEST_VAL_SPLIT*len(x_test))])
    y_test, y_val = np.split(y_test, [int(TEST_VAL_SPLIT*len(y_test))])

    if AUGMENTATION:
        print("starting augmentation")
        (x_load, y_load) = data_augmentation(x_train, y_train, x_test, y_test)
        print("augmentation done")
        label_count = [0] * NUM_CLASSES
        for lab in y_load:
            label_count[lab] += 1

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), label_count


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_val, y_val), (x_test, y_test), label_count = load_data(DATA_PATH, TO_SHUFFLE)

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
          test_img_idx = {}, train_test_split = {}, to_shuffle = {},
          augmentation = {}, label_count = {}, label_normalizer = {},
          save_augmentation_to_hdf5 = {}, learn rate = {}, train_all_layers = {},
          weights_to_use = {}, es_patience = {}, train_val_split = {},
          N_ENSEMBLE_MEMBERS = {}, MIN_DELTA = {}, Early_monitor = {}""".format(
              DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
              test_img_idx, TRAIN_TEST_SPLIT, TO_SHUFFLE,
              AUGMENTATION, label_count, LABEL_NORMALIZER,
              SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE, TRAIN_ALL_LAYERS,
              WEIGHTS_TO_USE, ES_PATIENCE, TRAIN_VAL_SPLIT,
              N_ENSEMBLE_MEMBERS, MIN_DELTA, EARLY_MONITOR))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return(x_train, y_train, x_val, y_val, x_test, y_test, test_img_idx)


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


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test, test_img_idx = prepare_data()

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

        (x_train, y_train) = shuffle_data(x_train, y_train)

        # Split train dataset so every enemble can train on an unique part of the data for maximum variance between the models
        x_train_splits = np.array_split(x_train, N_ENSEMBLE_MEMBERS)
        y_train_splits = np.array_split(y_train, N_ENSEMBLE_MEMBERS)

        ensemble = [fit_model(x_train_splits, y_train_splits, x_val, y_val, ensemble_model, log_dir, i, archi_name[ind]) for i in range(N_ENSEMBLE_MEMBERS)]

        os.remove('initial_weights.h5')


if __name__ == "__main__":
    main()
