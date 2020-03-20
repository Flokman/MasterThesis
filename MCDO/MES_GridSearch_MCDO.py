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
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
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
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3
DATASET_NAME = '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'

BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 500
AMOUNT_OF_PREDICTIONS = 50
MCDO_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = True
AUGMENTATION = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = True
ADD_DROPOUT = True
MCDO = True
TRAIN_ALL_LAYERS = True
DROPOUT_INSIDE = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001
ES_PATIENCE = 5
DROPOUTRATES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.5]

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
        MCDO_amount_of_predictions = {}, MCDO_batch_size = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        add_dropout_inside = {}, train_all_layers = {}, weights_to_use = {},
        mcdo = {}, es_patience = {}, train_val_split = {}, Dropoutrates: {}""".format(
        DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
        AMOUNT_OF_PREDICTIONS, MCDO_BATCH_SIZE, test_img_idx,
        TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION, label_count,
        LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
        DROPOUT_INSIDE, TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
        MCDO, ES_PATIENCE, TRAIN_VAL_SPLIT, DROPOUTRATES))

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


def get_dropout(input_tensor, prob=0.5):
    ''' Returns a dropout layer with probability prob, either trainable or not '''
    if MCDO:
        return Dropout(prob)(input_tensor, training=True)
    else:
        return Dropout(prob)(input_tensor)


def insert_intermediate_layer_in_keras(model, layer_id, prob):
    ''' Insert a dropout layer before the layer with layer_id '''
    inter_layers = [l for l in model.layers]

    x = inter_layers[0].output
    for i in range(1, len(inter_layers)):
        if i == layer_id:
            x = get_dropout(x, prob)
        x = inter_layers[i](x)

    new_model = Model(inputs=inter_layers[0].input, outputs=x)
    return new_model


def add_dropout(mcdo_model, dropoutrates=DROPOUTRATES):
    ''' Adds dropout layers either after all pool and dense layers or only after dense layers '''
    if DROPOUT_INSIDE:
        p_i = 0
        # Creating dictionary that maps layer names to the layers
        # layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])
        layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])

        for layer_name in layer_dict:
            layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])
            if layer_name.endswith('_pool'):
                layer_index = list(layer_dict).index(layer_name)
                # Add a dropout (trainable) layer
                mcdo_model = insert_intermediate_layer_in_keras(mcdo_model, layer_index + 1, dropoutrates[p_i])
                p_i += 1

        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcdo_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = get_dropout(x, dropoutrates[p_i])
        p_i += 1
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_dropout(x, dropoutrates[p_i])
        p_i += 1
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_dropout(x, dropoutrates[p_i])
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    else:
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcdo_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_dropout(x, 0.5)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_dropout(x, 0.5)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Creating new model
    mcdo_model = Model(inputs=all_layers[0].input, outputs=x)
    return mcdo_model


def create_model(dropoutrates=DROPOUTRATES):
    # VGG16 since it does not include batch normalization of dropout by itself
    MCDO_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if ADD_DROPOUT:
        MCDO_model = add_dropout(MCDO_model, dropoutrates)

    else:
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in MCDO_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

        # Creating new model
        MCDO_model = Model(inputs=all_layers[0].input, outputs=x)

    if TRAIN_ALL_LAYERS or DROPOUT_INSIDE:
        for layer in MCDO_model.layers:
            layer.trainable = True
    else:
        for layer in MCDO_model.layers[:-6]:
            layer.trainable = False
        for layer in MCDO_model.layers:
            print(layer, layer.trainable)

    MCDO_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)
    MCDO_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return MCDO_model


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    MCDO_model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=32, verbose=0)

    print("Start fitting monte carlo dropout model")

    X = x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))]
    X = X.astype('float32')
    Y = y_train[0:int(TRAIN_VAL_SPLIT*len(x_train))]
    X /= 255

    # define the grid search parameters
    dropoutrates = [[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.5],
                    [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6],
                    [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5]
                    ]
    param_grid = dict(dropoutrates=DROPOUTRATES)
    grid = GridSearchCV(estimator=MCDO_model, param_grid=param_grid, n_jobs=-1, cv=3)
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
