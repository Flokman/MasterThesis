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
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3
DATASET_NAME = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'

BATCH_SIZE = 64
NUM_CLASSES = 5
EPOCHS_1 = 150
ES_PATIENCE_1 = 20
EPOCHS_2 = 100
ES_PATIENCE_2 = 25
EPOCHS_3 = 5
ES_PATIENCE_3 = 5


TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.7 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.875
TEST_VAL_SPLIT = 0.66
SAVE_AUGMENTATION_TO_HDF5 = False

TO_SHUFFLE = False
AUGMENTATION = False
LABEL_NORMALIZER = False
TRAIN_ALL_LAYERS = True

WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001


# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + os.path.sep + 'Datasets' + DATASET_NAME


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
    # if to_shuffle:
    #     (x_load, y_load) = shuffle_data(x_load, y_load)

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

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs_1 = {},
        epochts_2 = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        train_all_layers = {}, weights_to_use = {},
        es_patience_1 = {}, es_patience_2 = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS_1,
            EPOCHS_2, test_img_idx,
            TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION, label_count,
            LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
            TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
            ES_PATIENCE_1, ES_PATIENCE_2, TRAIN_VAL_SPLIT))

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
    last_layer = Dense(4096, activation='relu', name='fc2')(x)
    classification = Dense(NUM_CLASSES, activation='softmax')(last_layer)
    Error = Dense(NUM_CLASSES, activation='softmax')(last_layer)

    out = concatenate([classification, Error])

    # Creating new model
    Error_model = Model(inputs=all_layers[0].input, outputs=out)

    Error_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)

    Error_model.compile(
        optimizer=optimizers.Adam(lr=0.000001),
        loss=Uncertainty_output(NUM_CLASSES).categorical_cross,
        metrics=['acc']
    )

    print("Start fitting")
    
    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), "MES" + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
                                   batch_size = BATCH_SIZE)
    
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
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_3)

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
    Uncertainty_output(NUM_CLASSES).results_if_label(Error_predictions, y_test, scatter=True, name='Messidor2_No_Conv')

    Error_predictions = Uncertainty_output(NUM_CLASSES).convert_output_to_uncertainty(Error_predictions)
    Uncertainty_output(NUM_CLASSES).results_if_label(Error_predictions, y_test, scatter=True, name='Messidor2')
    

    single_acc = accuracy_score(y_test.argmax(axis=1), Error_predictions[:, :NUM_CLASSES].argmax(axis=1))
    print(single_acc)
    dir_path_head_tail = os.path.split(os.path.dirname(os.getcwd()))
    if WEIGHTS_TO_USE != None:
        new_path = dir_path_head_tail[0] + os.path.sep + 'MES' + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + WEIGHTS_TO_USE + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    else:
        new_path = dir_path_head_tail[0] + os.path.sep + 'MES' + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    os.rename(fig_dir, new_path)


if __name__ == "__main__":
    main()
