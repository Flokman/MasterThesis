''' Returns the prediction and uncertainty of images in /test_images of a pretrained
(batch normarmalization) network of choice '''

import os
import random
import datetime
import glob
import re
import csv
import h5py
import tensorflow as tf
import numpy as np

# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils, backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Hyperparameters
DATANAME = 'POLAR'
METHODENAME = 'MCBN'

MCBN_PREDICTIONS = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
LEARN_RATE = 1

HDF5_DATASET = True
LABELS_AVAILABLE = False
TO_SHUFFLE = True
TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'


DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ONE_HIGHER_PATH = os.path.split(DIR_PATH_HEAD_TAIL[0])
DATA_ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
ROOT_PATH = ONE_HIGHER_PATH[0]

if DATANAME == 'MES':
    NUM_CLASSES = 5
    MINIBATCH_SIZE = 128
    MODEL_TO_USE = os.path.sep + METHODENAME
    MODEL_VERSION = os.path.sep + 'ImageNet_retrain_32B_144E_88A'
    MODEL_NAME = 'MCBN_model.h5'
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to
    DATASET_LOCATION = os.path.sep + 'Datasets'
    DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
    DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

if DATANAME == 'CIFAR10':
    from tensorflow.keras.datasets import cifar10
    NUM_CLASSES = 10
    MINIBATCH_SIZE = 128
    MODEL_TO_USE = os.path.sep + METHODENAME
    MODEL_VERSION = os.path.sep + 'CIFAR_ImageNet_Retrain_32B_57E_87A'
    MODEL_NAME = 'MCBN_model.h5'
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3 # target image size to resize to

if DATANAME == 'POLAR':
    NUM_CLASSES = 3
    MINIBATCH_SIZE = 16
    MODEL_TO_USE = os.path.sep + METHODENAME
    MODEL_VERSION = os.path.sep + '2020-03-22_10-50_imagenet_8B_51.9%A'
    MODEL_NAME = 'MCBN_model.h5'
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to
    DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
    DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


def shuffle_data(x_to_shuff, y_to_shuff):
    ''' Shuffle the data randomly '''
    combined = list(zip(x_to_shuff, y_to_shuff)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
 
    (x, y) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "x" then contains all the shuffled images and
                               # "y" contains all the shuffled labels.
    return (x, y)


def load_data(path, to_shuffle):
    '''' Load a dataset from a hdf5 file '''
    if DATANAME == 'POLAR':
        with h5py.File(path, "r") as f:
            (x_train_load, y_train_load, x_test_load, y_test_load) = np.array(f['x_train']), np.array(f['y_train']), np.array(f['x_test']), np.array(f['y_test'])
        train_label_count = [0] * NUM_CLASSES
        test_label_count = [0] * NUM_CLASSES
        for lab in y_train_load:
            train_label_count[lab] += 1
        for lab in y_test_load:
            test_label_count[lab] += 1

        if to_shuffle:
            (x_train_load, y_train_load) = shuffle_data(x_train_load, y_train_load)
            (x_test_load, y_test_load) = shuffle_data(x_test_load, y_test_load)

        return (x_train_load, y_train_load), (x_test_load, y_test_load), train_label_count, test_label_count
    
    if DATANAME == 'MES':
        '''' Load a dataset from a hdf5 file '''
        with h5py.File(path, "r") as f:
            (x_load, y_load) = np.array(f['x']), np.array(f['y'])
        label_count = [0] * NUM_CLASSES
        for lab in y_load:
            label_count[lab] += 1

        if to_shuffle:
            (x_load, y_load) = shuffle_data(x_load, y_load)

        # Divide the data into a train and test set
        x_train = x_load[0:int(TRAIN_TEST_SPLIT*len(x_load))]
        y_train = y_load[0:int(TRAIN_TEST_SPLIT*len(y_load))]

        x_test = x_load[int(TRAIN_TEST_SPLIT*len(x_load)):]
        y_test = y_load[int(TRAIN_TEST_SPLIT*len(y_load)):]

        return (x_train, y_train), (x_test, y_test), label_count


def load_hdf5_dataset():
    ''' Load a dataset, split and put in right format'''

    # Split the data between train and test sets
    if DATANAME == 'POLAR':
        (x_train, y_train), (x_test, y_test), train_label_count, test_label_count = load_data(DATA_PATH, TO_SHUFFLE)
    
    if DATANAME == 'MES':
        (x_train, y_train), (x_test, y_test), label_count = load_data(DATA_PATH, TO_SHUFFLE)

    if DATANAME == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
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

    return((x_train, y_train), (x_test, y_test))


def load_new_images():
    ''' Load, convert and resize the test images into numpy arrays '''
    images_path = os.getcwd() + TEST_IMAGES_LOCATION + os.path.sep + '*'
    print(images_path)

    # get all the image paths
    addrs = glob.glob(images_path)

    for i, addr in enumerate(addrs):
        if i % 1000 == 0 and i > 1:
            print('Image data: {}/{}'.format(i, len(addrs)))

        if i == 0:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            x = img_ar.reshape((1,) + img_ar.shape)
        
        else:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            img_ar = img_ar.reshape((1,) + img_ar.shape)
            x = np.vstack((x, img_ar))

    x = x.astype('float32')
    x /= 255
    print('x shape:', x.shape)
    print(x.shape[0], 'x samples')

    if LABELS_AVAILABLE:
        baseaddrs = []
        for im in addrs:
            if im.endswith('JPG'):
                im = im[:-3]
                im = im + 'jpg'
            baseaddrs.append(str(os.path.basename(im)))

        labels = []
        label_count = [0] * 5
        with open(TEST_IMAGES_LOCATION + TEST_IMAGES_LABELS_NAME + '.csv') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader) # skip header
            for row in reader:
                imaddrs = str(row[0])
                idx = 0
                suc = 2
                first = 0
                for im in baseaddrs:
                    if im == imaddrs:
                        if isinstance(row[1], str):
                            if row[1] == '': #if no class then 0
                                print(row[1])
                                row[1] = 0 
                            row[1] = int(float(row[1]))
                        labels.append(int(float(row[1])))
                        label_count[int(float(row[1]))] += 1
                        baseaddrs.pop(idx)
                        suc = 0
                    else:
                        if first == 0:
                            first = 1
                        if suc == 0:
                            suc = 0
                        else:
                            suc = 1
                    idx += 1
                if suc == 1:
                    print("failed:", imaddrs)
            
        print("label_count:", label_count)
        y.shape = (len(labels), len(labels))
        y[:,0] = np.arange(len(labels))
        y[0,:] = labels
        combined = list(zip(x, y)) # use zip() to bind the images and labels together    
        (x, y) = zip(*combined)
        # convert class vectors to binary class matrices
        y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
        
        return(x,y)
    else:
        return(x)


def create_minibatch(x, y):
    ''' Returns a minibatch of the train data '''
    combined = list(zip(x, y)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    # print("Random seed minibatch for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
    minibatch = combined[:MINIBATCH_SIZE]
    
    (x_minibatch, y_minibatch) = zip(*minibatch)  # *combined is used to separate all the tuples in the list combined,
                            # "x_minibatch" then contains all the shuffled images and
                            # "y_minibatch" contains all the shuffled labels.

    return(x_minibatch, y_minibatch)


def indepth_predictions(x_pred, y_pred, mc_predictions):
    ''' Creates more indepth predictions (print on command line and create figures in folder) '''
    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), METHODENAME + os.path.sep + DATANAME + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
    os.makedirs(fig_dir)
    log_dir = os.path.join(fig_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Dir to store Tensorboard data
    os.makedirs(log_dir)
    os.chdir(fig_dir)

    accs = []
    for y_p in mc_predictions:
        acc = accuracy_score(y_pred.argmax(axis=1), y_p.argmax(axis=1))
        accs.append(acc)
    print("MC accuracy: {:.1%}".format(sum(accs)/len(accs)))

    mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(y_pred.argmax(axis=1), mc_ensemble_pred)
    print("MC-ensemble accuracy: {:.1%}".format(ensemble_acc))

    confusion = tf.confusion_matrix(labels=y_pred.argmax(axis=1), predictions=mc_ensemble_pred, num_classes=NUM_CLASSES)
    sess = tf.Session()
    with sess.as_default():
        print(sess.run(confusion))

    plt.hist(accs)
    plt.axvline(x=ensemble_acc, color="b")
    plt.savefig('ensemble_acc.png')
    plt.clf()

    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in mc_predictions])
        print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        print("true label: {}".format(y_pred[i].argmax()))
        print()
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
            print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))


        x, y = list(range(len(p_0.mean(axis=0)))), p_0.mean(axis=0)
        plt.plot(x, y)
        plt.savefig('prob_var_' + str(i) + '.png')
        plt.clf()

        fig, axes = plt.subplots(5, 1, figsize=(12, 6))

        for i, ax in enumerate(fig.get_axes()):
            ax.hist(p_0[:, i], bins=100, range=(0, 1))
            ax.set_title(f"class {i}")
            ax.label_outer()

        fig.savefig('sub_plots' + str(i) + '.png', dpi=fig.dpi)


def main():
    ''' Main function '''
    # Get dataset
    if HDF5_DATASET:
        (x_train, y_train), (x_test, y_test) = load_hdf5_dataset()

    if LABELS_AVAILABLE:
        (x_train, y_train), (x_test, y_test) = load_hdf5_dataset()
        (x_test, y_test) = load_new_images()
    
    else:
        (x_train, y_train), (x_test, y_test) = load_hdf5_dataset()
        x_test = load_new_images()

    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
    print(os.getcwd())

    # Reload the model from the 2 files we saved
    with open('MCBN_model_config.json') as json_file:
        json_config = json_file.read()
    pre_trained_model = tf.keras.models.model_from_json(json_config)
    pre_trained_model.load_weights('MCBN_weights.h5')

    os.chdir(old_dir)

    # Set onoly batch normalization layers to trainable
    for layer in pre_trained_model.layers:
        if re.search('batch_normalization.*', layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
        print(layer.name, layer.trainable)

    adam = optimizers.Adam(lr=LEARN_RATE)
    pre_trained_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    pre_trained_model.summary()
    
    mcbn_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)

    org_model = pre_trained_model
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

    if HDF5_DATASET or LABELS_AVAILABLE:
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

    else:
        for i in range(len(x_test)):
            p_0 = np.array([p[i] for p in mcbn_predictions])
            print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
            # probability + variance
            for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))


if __name__== "__main__":
    main()

# POLAR:

    # Learn_rate = 0.001:
    # posterior mean: 2
    # class: 0; proba: 0.8%; var: 1.18%
    # class: 1; proba: 6.2%; var: 5.64%
    # class: 2; proba: 93.0%; var: 5.73%
    # posterior mean: 2
    # class: 0; proba: 24.8%; var: 19.64%
    # class: 1; proba: 0.0%; var: 0.01%
    # class: 2; proba: 75.2%; var: 19.63%
    # posterior mean: 2
    # class: 0; proba: 21.9%; var: 37.53%
    # class: 1; proba: 0.0%; var: 0.00%
    # class: 2; proba: 78.1%; var: 37.54%
    # posterior mean: 2
    # class: 0; proba: 0.3%; var: 0.45%
    # class: 1; proba: 49.2%; var: 38.04%
    # class: 2; proba: 50.5%; var: 37.75%
    # posterior mean: 2
    # class: 0; proba: 48.6%; var: 17.51%
    # class: 1; proba: 0.1%; var: 0.19%
    # class: 2; proba: 51.4%; var: 17.60%
    # posterior mean: 2
    # class: 0; proba: 0.0%; var: 0.01%
    # class: 1; proba: 4.5%; var: 8.70%
    # class: 2; proba: 95.5%; var: 8.70%

    # Learn_rate = 1:
    # posterior mean: 2
    # class: 0; proba: 27.5%; var: 42.48%
    # class: 1; proba: 3.1%; var: 16.02%
    # class: 2; proba: 69.4%; var: 43.77%
    # posterior mean: 2
    # class: 0; proba: 10.6%; var: 28.36%
    # class: 1; proba: 6.4%; var: 23.61%
    # class: 2; proba: 83.0%; var: 35.04%
    # posterior mean: 2
    # class: 0; proba: 27.3%; var: 42.68%
    # class: 1; proba: 3.3%; var: 17.57%
    # class: 2; proba: 69.5%; var: 44.32%
    # posterior mean: 0
    # class: 0; proba: 48.7%; var: 48.03%
    # class: 1; proba: 10.5%; var: 29.80%
    # class: 2; proba: 40.8%; var: 47.32%
    # posterior mean: 2
    # class: 0; proba: 12.5%; var: 31.70%
    # class: 1; proba: 1.6%; var: 11.13%
    # class: 2; proba: 85.9%; var: 33.32%
    # posterior mean: 2
    # class: 0; proba: 15.9%; var: 34.35%
    # class: 1; proba: 28.2%; var: 43.01%
    # class: 2; proba: 55.9%; var: 47.64%
