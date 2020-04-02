''' Returns the prediction and uncertainty of images in /test_images of all pretrained
networks '''

import os
import random
import datetime
import glob
import re
import csv
import h5py
import tensorflow as tf
import numpy as np
from statistics import mean

# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils, backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")



import time
from functools import wraps

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print ("Function %s called %d times. " % (fname, data[0]),)
        print ('Execution time max: %.3f, average: %.3f' % (max_time, avg_time))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}



# Hyperparameters
DATANAME = 'POLAR'
METHODENAMES = ['MCDO', 'MCBN', 'Ensemble', 'VarianceOutput']

TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test

TEST_ON_OWN_AND_NEW_DATASET = True
TEST_ON_OWN_DATASET = False
TEST_ON_NEW_DATASET = False
LABELS_AVAILABLE = False
TO_SHUFFLE = False
TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'

if (DATANAME == 'MES' or DATANAME == 'POLAR') and TEST_ON_OWN_AND_NEW_DATASET == True:
    TEST_ON_OWN_DATASET = True


if DATANAME == 'MES':
    # Hyperparameters Messidor
    NUM_CLASSES = 5
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to


if DATANAME == 'CIFAR10':
    # Hyperparameters Cifar
    NUM_CLASSES = 10
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3 # target image size to resize to


if DATANAME == 'POLAR':
    # Hyperparameters Polar
    NUM_CLASSES = 3
    IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to


DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ONE_HIGHER_PATH = os.path.split(DIR_PATH_HEAD_TAIL[0])
DATA_ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
ROOT_PATH = ONE_HIGHER_PATH[0]


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


def load_hdf5_dataset(DATA_PATH):
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

    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return((x_train, y_train), (x_test, y_test))


def load_new_images():
    ''' Load, convert and resize the test images into numpy arrays '''
    images_path = os.getcwd() + TEST_IMAGES_LOCATION + os.path.sep + '*'
    # print(images_path)

    # get all the image paths
    addrs = glob.glob(images_path)

    for i, addr in enumerate(addrs):
        if i % 1000 == 0 and i > 1:
            print('Image data: {}/{}'.format(i, len(addrs)))

        if i == 0:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            x_pred = img_ar.reshape((1,) + img_ar.shape)

        else:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            img_ar = img_ar.reshape((1,) + img_ar.shape)
            x_pred = np.vstack((x_pred, img_ar))

    x_pred = x_pred.astype('float32')
    x_pred /= 255
    # print('x_pred shape:', x_pred.shape)
    # print(x_pred.shape[0], 'x_pred samples')

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
        y_pred.shape = (len(labels), len(labels))
        y_pred[:, 0] = np.arange(len(labels))
        y_pred[0, :] = labels
        combined = list(zip(x_pred, y_pred)) # use zip() to bind the images and labels together
        (x_pred, y_pred) = zip(*combined)
        # convert class vectors to binary class matrices
        y_pred = tf.keras.utils.to_categorical(y_pred, NUM_CLASSES)

        return(x_pred, y_pred)
    else:
        return x_pred


def indepth_predictions(x_pred, y_pred, mc_predictions, METHODENAME):
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


def test_on_own_func(methodname, predictions, y_test):
    # score of the model
    # accs = []
    # for y_p in predictions:
    #     acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
    #     accs.append(acc)
    # print("{} accuracy: {:.1%}".format(methodname, sum(accs)/len(accs)))

    pred = np.array(predictions).mean(axis=0).argmax(axis=1)
    acc = accuracy_score(y_test.argmax(axis=1), pred)
    print("{} combined accuracy: {:.1%}".format(methodname, acc))

    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=pred,
                                    num_classes=NUM_CLASSES)
    print(confusion)

    correct_var = []
    wrong_var = []
    for i in range(len(y_test)):
        p_0 = np.array([p[i] for p in predictions])
        # print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        
        # probability + variance
        correct_ind = y_test[i].argmax()
        predicted_ind = p_0.mean(axis=0).argmax() #TODO test if this is returning the argmax of all predicitons
        correct_pred = False 
        if correct_ind == predicted_ind:
            correct_pred = True
        # for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
        #     # print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))
        #     if correct_pred and l == correct_ind:
        #         correct_var.append(var)
        #     else:
        #         wrong_var.append(var)

        if correct_pred:
            for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                # print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))
                if l == correct_ind:
                    correct_var.append(var)
        else:
             for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                # print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))
                if l == predicted_ind:
                    wrong_var.append(var)           

    print("")
    print("Uncertainty on original test dataset when correctly predicted = {:.2%}".format(mean(correct_var))) 
    print("Uncertainty on original test dataset when wrongly predicted = {:.2%}".format(mean(wrong_var)))    


def test_on_new_func(new_images_predictions, x_pred, more_info = False):
    total_new_var = []
    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in new_images_predictions])
        # print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
            # print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))
            total_new_var.append(var)

    print("Mean uncertainty on predicted class of new data = {:.2%}".format(mean(total_new_var)))
    
    if more_info:
        for i in range(len(x_pred)):
            p_0 = np.array([p[i] for p in new_images_predictions])
            print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
            # probability + variance
            for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))


@profile
def MCDO(METHODENAME):

    MCDO_PREDICTIONS = 250

    if DATANAME == 'MES':
        MCDO_BATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + 'MES_ImageNet_Retrain_32B_93E_52A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + 'CIFAR_ImageNet_retrain_32B_95E_86A'
        MODEL_NAME = 'MCDO_model.h5'

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + '2020-03-22_10-56_imagenet_16B_61.1%A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    def mcdo_predict(pre_trained_model, x_pred):
        mcdo_predictions = []
        # progress_bar = tf.keras.utils.Progbar(target=MCDO_PREDICTIONS, interval=5)
        for i in range(MCDO_PREDICTIONS):
            # progress_bar.update(i)
            y_p = pre_trained_model.predict(x_pred, batch_size=MCDO_BATCH_SIZE)
            mcdo_predictions.append(y_p)
        return mcdo_predictions

    def method_main():
        ''' Main function '''
        # Get dataset
        if TEST_ON_OWN_DATASET and not TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)

        elif LABELS_AVAILABLE:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            (x_pred, y_pred) = load_new_images()

        elif TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()           

        elif TEST_ON_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()

        old_dir = os.getcwd()
        os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
        print(os.getcwd())

        # Reload the model from the 2 files we saved
        with open('MCDO_model_config.json') as json_file:
            json_config = json_file.read()
        pre_trained_model = tf.keras.models.model_from_json(json_config)
        pre_trained_model.load_weights('MCDO_weights.h5')

        # pre_trained_model.summary()

        os.chdir(old_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            mcdo_predictions = mcdo_predict(pre_trained_model, x_test)
            test_on_own_func(METHODENAME, mcdo_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            mcdo_new_images_predictions = mcdo_predict(pre_trained_model, x_pred)
            test_on_new_func(mcdo_new_images_predictions, x_pred)
        
        else:
            mcdo_new_images_predictions = mcdo_predict(pre_trained_model, x_pred)
            test_on_new_func(mcdo_new_images_predictions, x_pred, more_info = True)


    method_main()


@profile
def MCBN(METHODENAME):
    MCBN_PREDICTIONS = 250
    LEARN_RATE = 1

    if DATANAME == 'MES':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + 'ImageNet_retrain_32B_144E_88A'
        MODEL_NAME = 'MCBN_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + 'CIFAR_ImageNet_Retrain_32B_57E_87A'
        MODEL_NAME = 'MCBN_model.h5'

    if DATANAME == 'POLAR':
        MINIBATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODENAME
        MODEL_VERSION = os.path.sep + '2020-03-22_10-50_imagenet_8B_51.9%A'
        MODEL_NAME = 'MCBN_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


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


    def mcbn_predict(pre_trained_model, x_train, y_train, x_pred):
        mcbn_predictions = []
        # progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)

        org_model = pre_trained_model
        for i in range(MCBN_PREDICTIONS):
            # progress_bar.update(i)
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

            y_p = MCBN_model.predict(x_pred, batch_size=len(x_pred)) #Predict for bn look at (sigma and mu only one to chance, not the others)
            mcbn_predictions.append(y_p)
        
        return mcbn_predictions


    def method_main():
        ''' Main function '''
        # Get dataset
        if TEST_ON_OWN_DATASET and not TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)

        elif LABELS_AVAILABLE:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            (x_pred, y_pred) = load_new_images()

        elif TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()           

        elif TEST_ON_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()

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
            # print(layer.name, layer.trainable)

        adam = optimizers.Adam(lr=LEARN_RATE)
        pre_trained_model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # pre_trained_model.summary()  
   
        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            mcbn_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_test)
            test_on_own_func(METHODENAME, mcbn_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            mcbn_new_images_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_pred)
            test_on_new_func(mcbn_new_images_predictions, x_pred)
        
        else:
            mcbn_new_images_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_pred)
            test_on_new_func(mcbn_new_images_predictions, x_pred, more_info = True)


    method_main()


@profile
def Ensemble(METHODENAME):
    if DATANAME == 'MES':
        # Hyperparameters Messidor
        N_FOLDERS = 2
        N_ENSEMBLE_MEMBERS = [43, 27]
        MODEL_VERSION = ['/MES_32B_43EN', '/MES_ImageNet_32B_27EN']
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5


    if DATANAME == 'CIFAR10':
        # Hyperparameters Cifar
        N_FOLDERS = 1
        # N_ENSEMBLE_MEMBERS = [20, 20]
        N_ENSEMBLE_MEMBERS = [40]
        # MODEL_VERSION = ['CIF_ImageNet_32B_20EN', 'CIF_ImageNet_32B_20EN_2']
        MODEL_VERSION = ['2020-03-19_16-19-18']


    if DATANAME == 'POLAR':
        # Hyperparameters Polar
        N_FOLDERS = 1
        N_ENSEMBLE_MEMBERS = [40]
        MODEL_VERSION = ['2020-03-30_09-58_imagenet_16B_61.1%A']
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    MODEL_TO_USE = os.path.sep + METHODENAME


    def ensemble_predict(x_pred):
        mc_predictions = []
        # progress_bar = tf.keras.utils.Progbar(target=sum(N_ENSEMBLE_MEMBERS), interval=5)
        for i in range(N_FOLDERS):
            old_dir = os.getcwd()
            os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + os.path.sep + MODEL_VERSION[i] + os.path.sep)
            with open('ensemble_model_config_0.json') as json_file:
                json_config = json_file.read()
            pre_trained_model = tf.keras.models.model_from_json(json_config)
            os.chdir(old_dir)

            for j in range(N_ENSEMBLE_MEMBERS[i]):
                old_dir = os.getcwd()
                os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + os.path.sep + MODEL_VERSION[i] + os.path.sep)
                # print(os.getcwd())

                # Reload the model from the 2 files we saved

                pre_trained_model.load_weights('ensemble_weights_{}.h5'.format(j))
                # pre_trained_model.summary()
                os.chdir(old_dir)

                # progress_bar.update(j)
                y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred))
                # print(y_p)
                mc_predictions.append(y_p)
            K.clear_session()
        
        return mc_predictions 


    def method_main():
        ''' Main function '''

        print("Dataset_name: {}, N_folders: {}, total mebers: {}".format(DATANAME, N_FOLDERS, sum(N_ENSEMBLE_MEMBERS)))

        # Get dataset
        if TEST_ON_OWN_DATASET and not TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)

        elif LABELS_AVAILABLE:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            (x_pred, y_pred) = load_new_images()

        elif TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()           

        elif TEST_ON_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()


        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            ensemble_predictions = ensemble_predict(x_test)
            test_on_own_func(METHODENAME, ensemble_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            ensemble_new_images_predictions = ensemble_predict(x_pred)
            test_on_new_func(ensemble_new_images_predictions, x_pred)
        
        else:
            ensemble_only_new_images_predictions = ensemble_predict(x_pred)
            test_on_new_func(ensemble_new_images_predictions, x_pred, more_info = True)


    method_main()
    # TODO: save testimage with in title predicted class, acc and prob


@profile
def VarianceOutput(METHODENAME):
    # Hyperparameters
    MODEL_TO_USE = os.path.sep + METHODENAME

    if DATANAME == 'MES':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + 'MES_ImageNet_Retrain_32B_90E_52A'
        MODEL_NAME = 'variance_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + 'CIF_ImageNet_32B_95E_68A'
        MODEL_NAME = 'variance_model.h5'

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_VERSION = os.path.sep + '2020-03-29_10-41_imagenet_16B_64.8%A'
        MODEL_NAME = 'variance_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    def convert_to_var(prediction, y_test, label_avail = True):
        #Convert var pred to uncertainty
        if label_avail:
            for ind, pred in enumerate(prediction):
                classif = pred[:NUM_CLASSES]
                var = np.abs(pred[NUM_CLASSES:])

                for i in range(0, NUM_CLASSES):
                    pred_error = var[i]
                    true_error = pow((classif[i] - y_test[ind][i]), 2)
                    var[i] = abs(true_error - pred_error)
                prediction[ind][NUM_CLASSES:] = var

        else:
            for ind, pred in enumerate(prediction):
                classif = pred[:NUM_CLASSES]
                classif_ind = np.argmax(classif)
                var = np.abs(pred[NUM_CLASSES:])

                for i in range(0, NUM_CLASSES):
                    pred_var = var[i]
                    if i == classif_ind:
                        # Highest predicted class, so error as if true
                        true_error = pow((classif[i] - 1), 2)
                    else:
                        # Not highest predicted class, so error as if false
                        true_error = pow((classif[i]), 2)
                    var[i] = abs(true_error - pred_var)
                prediction[ind][NUM_CLASSES:] = var            

        return prediction


    def var_label(org_data_prediction, y_test):
        # score on the test images (if label avaialable)
        true_labels = [np.argmax(i) for i in y_test]
        wrong = 0
        correct = 0
        correct_var = []
        wrong_var = []

        for ind, pred in enumerate(org_data_prediction):
            true_label = true_labels[ind]
            classif = pred[:NUM_CLASSES]
            classif_ind = np.argmax(classif)
            var = pred[NUM_CLASSES:]
            var_wrong = var[classif_ind]
            var_correct = var[true_label]

            if classif_ind != true_label:
                wrong += 1
                wrong_var.append(var[classif_ind])
                # print("Pred: {}, true: {}".format(classif_ind, true_label))
                # print("Var_pred: {}, var_true: {}".format(var_wrong, var_correct))
            if classif_ind == true_label:
                correct += 1
                correct_var.append(var[true_label])

        acc = accuracy_score(y_test.argmax(axis=1), org_data_prediction[:, :NUM_CLASSES].argmax(axis=1))
        print("Accuracy on original test dataset: {:.1%}".format(acc))

        confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=org_data_prediction[:, :NUM_CLASSES].argmax(axis=1),
                                        num_classes=NUM_CLASSES)
        print(confusion)

        print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, 100- (wrong/correct)*100))
        print("")
        print("Uncertainty on original test dataset when correctly predicted = {:.2%}".format(mean(correct_var))) 
        print("Uncertainty on original test dataset when wrongly predicted = {:.2%}".format(mean(wrong_var)))    


    def var_no_label(new_images_predictions, more_info = False):
        new_var_pred = []
        new_var_not_pred = []

        for ind, pred in enumerate(new_images_predictions):
            classif = pred[:NUM_CLASSES]
            classif_ind = np.argmax(classif)
            var = pred[NUM_CLASSES:]
            var_pred = var[classif_ind]
            new_var_pred.append(var_pred)
            
            for i in range(0, NUM_CLASSES):
                if classif_ind != i:
                    new_var_not_pred.append(var[i])


        print("Mean uncertainty on predicted class of new data = {:.2%}".format(mean(new_var_pred)))
        print("Mean uncertainty on not predicted classes of new data = {:.2%}".format(mean(new_var_not_pred)))

        if more_info:
            for ind, pred in enumerate(new_images_predictions):
                classif = pred[:NUM_CLASSES]
                classif_ind = np.argmax(classif)
                var = abs(pred[NUM_CLASSES:])
                if_true_error = pow((classif[classif_ind] - 1), 2)
                var_pred = abs(if_true_error - var[classif_ind])
                print("Predicted class: {}, Uncertainty of prediction: {:.2%}".format(classif_ind, var_pred))


    def method_main():
        ''' Main function '''
        # Get dataset
        if TEST_ON_OWN_DATASET and not TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)

        elif LABELS_AVAILABLE:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            (x_pred, y_pred) = load_new_images()

        elif TEST_ON_OWN_AND_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()           

        elif TEST_ON_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            x_pred = load_new_images()

        old_dir = os.getcwd()
        os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
        print(os.getcwd())

        # Reload the model from the 2 files we saved
        with open('variance_model_config.json') as json_file:
            json_config = json_file.read()
        pre_trained_model = tf.keras.models.model_from_json(json_config)
        pre_trained_model.load_weights('variance_weights.h5')
        # pre_trained_model.summary()
        os.chdir(old_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            variance_org_dataset = pre_trained_model.predict(x_test)
            variance_org_dataset = convert_to_var(variance_org_dataset, y_test)
            var_label(variance_org_dataset, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            variance_new_images_predictions = pre_trained_model.predict(x_pred)
            variance_new_images_predictions = convert_to_var(variance_new_images_predictions, y_test = None, label_avail=False)
            var_no_label(variance_new_images_predictions)
        
        else:
            variance_new_images_predictions = pre_trained_model.predict(x_pred)
            variance_new_images_predictions = convert_to_var(variance_new_images_predictions, y_test = None, label_avail=False)
            var_no_label(variance_new_images_predictions, more_info = True)


    method_main()


@profile
def main():
    for method_ind, METHODENAME in enumerate(METHODENAMES):
        if METHODENAME == 'MCDO':
            print('')
            print("START MCDO")
            MCDO(METHODENAME)
        if METHODENAME == 'MCBN':
            print('')
            print("START MCBN")
            MCBN(METHODENAME)
        if METHODENAME == 'Ensemble':
            print('')
            print("START Ensemble")
            Ensemble(METHODENAME)
        if METHODENAME == 'VarianceOutput':
            print('')
            print("START VarianceOutput")
            VarianceOutput(METHODENAME)

if __name__ == "__main__":
    main()
    print('')
    print("#################################")
    print_prof_data()
    print("#################################")


# TODO make testing on own dataset work
# TODO test on bullshit, and some real ones for comparison
# TODO for varianceoutput train variance on different portion of data (different then for first loss)
# Truckje: laatste laag VGG16 opblazen () als output verandert betekent dat data niet gebalanceerd is
# Wanneer niet veranderd, dan overfitting
# x = Dense(4096, activation='relu', name='fc1')(x) -> 4096*2 of meer