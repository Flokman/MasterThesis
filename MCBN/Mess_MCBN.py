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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

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
        MCBN_PREDICTIONS = {}, Mini_batch_size = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        add_bn_inside = {}, train_all_layers = {}, weights_to_use = {},
        es_patience = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
            MCBN_PREDICTIONS, MINIBATCH_SIZE, test_img_idx,
            TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION, label_count,
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


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

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

    adam = optimizers.Adam(lr=LEARN_RATE)
    MCBN_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Start fitting monte carlo batch_normalization model")

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), "MES" + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode='auto', verbose=1, patience=ES_PATIENCE)

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size = BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)


    MCBN_model.fit(train_generator,
                   epochs=EPOCHS,
                   verbose=2,
                   validation_data=val_generator,
                   callbacks=[tensorboard_callback, early_stopping])

    # Save JSON config to disk
    json_config = MCBN_model.to_json()
    with open('MCBN_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    MCBN_model.save_weights('MCBN_weights_.h5')

    # Set onoly batch normalization layers to trainable
    for layer in MCBN_model.layers:
        if re.search('batch_normalization.*', layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
        print(layer.name, layer.trainable)

    adam = optimizers.Adam(lr=LEARN_RATE)
    MCBN_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    mcbn_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)
    
    
    org_model = MCBN_model
    for i in range(MCBN_PREDICTIONS):
        progress_bar.update(i)
        # Create new random minibatch from train data
        x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
        x_minibatch = np.asarray(x_minibatch)
        y_minibatch = np.asarray(y_minibatch)

        datagen = ImageDataGenerator(rescale=1./255)
        minibatch_generator = datagen.flow(x_minibatch,
                                    y_minibatch,
                                    batch_size = MINIBATCH_SIZE)


        # Fit the BN layers with the new minibatch, leave all other weights the same
        MCBN_model = org_model
        MCBN_model.fit(minibatch_generator,
                            epochs=1,
                            verbose=0)

        y_p = MCBN_model.predict(x_test, batch_size=len(x_test)) #Predict for bn look at (sigma and mu only one to chance, not the others)
        mcbn_predictions.append(y_p)

    # score of the MCBN model
    accs = []
    for y_p in mcbn_predictions:
        acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
        accs.append(acc)
    print("MCBN accuracy: {:.1%}".format(sum(accs)/len(accs)))

    mcbn_ensemble_pred = np.array(mcbn_predictions).mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(y_test.argmax(axis=1), mcbn_ensemble_pred)
    print("MCBN-ensemble accuracy: {:.1%}".format(ensemble_acc))

    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mcbn_ensemble_pred,
                                    num_classes=NUM_CLASSES)
    print(confusion)

    plt.hist(accs)
    plt.axvline(x=ensemble_acc, color="b")
    plt.savefig('ensemble_acc.png')
    plt.clf()

    plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])

    p_0 = np.array([p[test_img_idx] for p in mcbn_predictions])
    print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
    print("true label: {}".format(y_test[test_img_idx].argmax()))
    print()
    # probability + variance
    for i, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
        print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))

    x_axis, y_axis = list(range(len(p_0.mean(axis=0)))), p_0.mean(axis=0)
    plt.plot(x_axis, y_axis)
    plt.savefig('prob_var_' + str(test_img_idx) + '.png')
    plt.clf()

    fig = plt.subplots(NUM_CLASSES, 1, figsize=(12, 6))[0]

    for i, ax in enumerate(fig.get_axes()):
        ax.hist(p_0[:, i], bins=100, range=(0, 1))
        ax.set_title(f"class {i}")
        ax.label_outer()

    fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)

if __name__ == "__main__":
    main()
