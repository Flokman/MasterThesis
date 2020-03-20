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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

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


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
        MCDO_amount_of_predictions = {}, MCDO_batch_size = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        add_dropout_inside = {}, train_all_layers = {}, weights_to_use = {},
        mcdo = {}, es_patience = {}, train_val_split = {}, Dropoutrates: {}""".format(
        DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
        AMOUNT_OF_PREDICTIONS, MCDO_BATCH_SIZE, test_img_idx,
        TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION,
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


def add_dropout(mcdo_model):
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
                mcdo_model = insert_intermediate_layer_in_keras(mcdo_model, layer_index + 1, DROPOUTRATES[p_i])
                p_i += 1

        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcdo_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = get_dropout(x, DROPOUTRATES[p_i])
        p_i += 1
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_dropout(x, DROPOUTRATES[p_i])
        p_i += 1
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_dropout(x, DROPOUTRATES[p_i])
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


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    # VGG16 since it does not include batch normalization of dropout by itself
    MCDO_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if ADD_DROPOUT:
        MCDO_model = add_dropout(MCDO_model)

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

    print("Start fitting monte carlo dropout model")

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), "CIFAR10" + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
    
    MCDO_model.fit(train_generator,
                   epochs=EPOCHS,
                   verbose=2,
                   validation_data=val_generator,
                   callbacks=[tensorboard_callback, early_stopping])

    # Save JSON config to disk
    json_config = MCDO_model.to_json()
    with open('MCDO_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    MCDO_model.save_weights('MCDO_weights_.h5')

    mcdo_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=AMOUNT_OF_PREDICTIONS, interval=5)
    for i in range(AMOUNT_OF_PREDICTIONS):
        progress_bar.update(i)
        y_p = MCDO_model.predict(x_test, batch_size=MCDO_BATCH_SIZE)
        mcdo_predictions.append(y_p)

    # score of the MCDO model
    accs = []
    for y_p in mcdo_predictions:
        acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
        accs.append(acc)
    print("MCDO accuracy: {:.1%}".format(sum(accs)/len(accs)))

    mcdo_ensemble_pred = np.array(mcdo_predictions).mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(y_test.argmax(axis=1), mcdo_ensemble_pred)
    print("MCDO-ensemble accuracy: {:.1%}".format(ensemble_acc))

    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mcdo_ensemble_pred,
                                    num_classes=NUM_CLASSES)
    print(confusion)

    plt.hist(accs)
    plt.axvline(x=ensemble_acc, color="b")
    plt.savefig('ensemble_acc.png')
    plt.clf()

    plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])

    p_0 = np.array([p[test_img_idx] for p in mcdo_predictions])
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
