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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
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

BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 500
N_ENSEMBLE_MEMBERS = 40
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
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return(x_train, y_train, x_test, y_test, test_img_idx)


def fit_model(x_train_splits, y_train_splits, x_val, y_val, ensemble_model, log_dir, i):
    ensemble_model.load_weights('initial_weights.h5')

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
    with open("ensemble_model_config_{}.json".format(i), 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    ensemble_model.save_weights("ensemble_weights_{}.h5".format(i))

    # To save memory, clear memory and return something else
    K.clear_session()

    return 1


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
    ensemble_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    # Stacking a new simple convolutional network on top of vgg16
    all_layers = [l for l in ensemble_model.layers]
    x = all_layers[0].output
    for i in range(1, len(all_layers)):
        x = all_layers[i](x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Creating new model
    ensemble_model = Model(inputs=all_layers[0].input, outputs=x)

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

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    # Save initial weights
    ensemble_model.save_weights('initial_weights.h5')

    # Split train dataset so every enemble can train on an unique part of the data for maximum variance between the models
    x_train_splits = np.split(x_train, N_ENSEMBLE_MEMBERS)
    y_train_splits = np.split(y_train, N_ENSEMBLE_MEMBERS)

    print("Length split part: ", len(x_train_splits[i]))

    ensemble = [fit_model(x_train_splits, y_train_splits, x_val, y_val, ensemble_model, log_dir, i) for i in range(N_ENSEMBLE_MEMBERS)]

    os.remove('initial_weights.h5')

    # ensemble_predictions = [model.predict(x_test, batch_size=TEST_BATCH_SIZE) for model in ensemble]
    # # ensemble_predictions = array(ensemble_predictions)

    # # score of the MCDO model
    # accs = []
    # for y_p in ensemble_predictions:
    #     acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
    #     accs.append(acc)
    # print("Highest acc of model in ensemble: {:.1%}".format(sum(accs)/len(accs)))

    # ensemble_pred = np.array(ensemble_predictions).mean(axis=0).argmax(axis=1)
    # ensemble_acc = accuracy_score(y_test.argmax(axis=1), ensemble_pred)
    # print("Mean ensemble accuracy: {:.1%}".format(ensemble_acc))

    # dir_path_head_tail = os.path.split(os.path.dirname(os.getcwd()))
    # new_path = dir_path_head_tail[0] + os.path.sep + RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + WEIGHTS_TO_USE + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(ensemble_acc)
    # os.rename(fig_dir, new_path)

    # confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=ensemble_pred,
    #                                 num_classes=NUM_CLASSES)
    # print(confusion)

    # plt.hist(accs)
    # plt.axvline(x=ensemble_acc, color="b")
    # plt.savefig('ensemble_acc.png')
    # plt.clf()

    # plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])

    # p_0 = np.array([p[test_img_idx] for p in ensemble_predictions])
    # print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
    # print("true label: {}".format(y_test[test_img_idx].argmax()))
    # print()
    # # probability + variance
    # for i, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
    #     print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))

    # x_axis, y_axis = list(range(len(p_0.mean(axis=0)))), p_0.mean(axis=0)
    # plt.plot(x_axis, y_axis)
    # plt.savefig('prob_var_' + str(test_img_idx) + '.png')
    # plt.clf()

    # fig = plt.subplots(NUM_CLASSES, 1, figsize=(12, 6))[0]

    # for i, ax in enumerate(fig.get_axes()):
    #     ax.hist(p_0[:, i], bins=100, range=(0, 1))
    #     ax.set_title(f"class {i}")
    #     ax.label_outer()

    # fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)


if __name__ == "__main__":
    main()
