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

# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from statistics import mean

from PIL import Image

from multiprocessing import Process, Queue

from tensorflow.keras import Input, layers, models, utils, backend
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from uncertainty_output import Uncertainty_output

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
DATANAME = 'MES'
NEW_DATA = 'MNIST' # or 'local' for local folder
METHODNAMES = ['MCDO', 'MCBN', 'Ensemble', 'VarianceOutput']
# METHODNAMES = ['VarianceOutput']

TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test

TEST_ON_OWN_AND_NEW_DATASET = False
TEST_ON_OWN_DATASET = True
TEST_ON_NEW_DATASET = False
LABELS_AVAILABLE = False
TO_SHUFFLE = False
TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'
HOME_DIR = os.getcwd()

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


def load_hdf5_dataset(DATA_PATH, new_data=False):
    ''' Load a dataset, split and put in right format'''

    # Split the data between train and test sets
    if not new_data:
        if DATANAME == 'POLAR':
            (x_train, y_train), (x_test, y_test), train_label_count, test_label_count = load_data(DATA_PATH, TO_SHUFFLE)

        if DATANAME == 'MES':
            (x_train, y_train), (x_test, y_test), label_count = load_data(DATA_PATH, TO_SHUFFLE)

        if DATANAME == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    else:
        if NEW_DATA == 'MNIST':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

    
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
    if NEW_DATA == 'local':
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
        x_pred = np.asarray(x_pred)
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

    elif NEW_DATA == 'MNIST':
        (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(None, new_data=True)

        for ind, img_ar in enumerate(x_test):
            if ind == 0:
                img = Image.fromarray(img_ar)
                img = img.resize((32, 32))
                img_ar = np.array(img)
                img_ar = np.stack((img_ar, img_ar, img_ar), axis=2)
                x_pred = img_ar.reshape((1,) + img_ar.shape)
            else:    
                img = Image.fromarray(img_ar)
                img = img.resize((32, 32))
                img_ar = np.array(img)
                img_ar = np.stack((img_ar, img_ar, img_ar), axis=2)
                img_ar = img_ar.reshape((1,) + img_ar.shape)
                x_pred = np.vstack((x_pred, img_ar))
        
        x_pred = x_pred.astype('float32')
        x_pred = np.asarray(x_pred)

        if LABELS_AVAILABLE:
            # y_test = np.asarray(y_test)
            # y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
            return x_pred, y_test
        else:
            return x_pred


def indepth_predictions(x_pred, y_pred, mc_predictions, METHODNAME):
    ''' Creates more indepth predictions (print on command line and create figures in folder) '''
    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), METHODNAME + os.path.sep + DATANAME + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) 
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
    mean_predictions_all = np.array(predictions).mean(axis=0).argmax(axis=1)
    acc = accuracy_score(y_test.argmax(axis=1), mean_predictions_all)
    print("{} combined accuracy: {:.1%}".format(methodname, acc))

    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mean_predictions_all,
                                    num_classes=NUM_CLASSES)
    print(confusion)


    true_labels = [np.argmax(i) for i in y_test]
    wrong = 0
    correct = 0
    # Only when correctly predict, info of true class
    correct_unc = []
    correct_prob = []
    # Only when wrongly predicted, info of highest wrong pred
    high_wrong_unc = []
    high_wrong_prob = []
    # Only when wrongly predicted, info of true class
    true_wrong_unc = []
    true_wrong_prob = []        
    # Info of all incorrect classes
    all_wrong_unc = []
    all_wrong_prob = []
    # Info of all not true label classes
    not_true_label_unc = []
    not_true_label_prob = []
    # Info of all classes
    all_probabilities = []
    all_uncertainties = []

    for ind in range(len(y_test)):
        all_predictions_single = np.array([p[ind] for p in predictions])
        # print("posterior mean: {}".format(all_predictions_single.mean(axis=0).argmax()))
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = all_predictions_single.mean(axis=0).argmax()

        for l, (prob, var) in enumerate(zip(all_predictions_single.mean(axis=0), all_predictions_single.std(axis=0))):
            all_probabilities.append(prob)
            all_uncertainties.append(var)

            if l == true_label:
                if highest_pred_ind == true_label:
                    correct += 1
                    correct_unc.append(var)
                    correct_prob.append(prob)

                else:
                    wrong += 1
                    true_wrong_unc.append(var)
                    true_wrong_prob.append(prob)

                    all_wrong_unc.append(var)
                    all_wrong_prob.append(prob)
            
            if l == highest_pred_ind:
                high_wrong_unc.append(var)
                high_wrong_prob.append(prob)

            else:
                all_wrong_unc.append(var)
                all_wrong_prob.append(prob)

                not_true_label_unc.append(var)
                not_true_label_prob.append(prob)  



    print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(correct+wrong))*100))
    print("")
    print("Mean probability on true label of {} test dataset when correctly predicted = {:.2%}".format(DATANAME, mean(correct_prob)))
    print("Mean uncertainty on true label of {} test dataset when correctly predicted = {:.2%}".format(DATANAME, mean(correct_unc)))
    print("Mean probability on true label of {} test dataset when wrongly predicted = {:.2%}".format(DATANAME, mean(true_wrong_prob))) 
    print("Mean uncertainty on true label of {} test dataset when wrongly predicted = {:.2%}".format(DATANAME, mean(true_wrong_unc)))    

    print("")
    print("Mean probability on highest predicted on {} test dataset when wrong = {:.2%}".format(DATANAME, mean(high_wrong_prob))) 
    print("Mean uncertainty on highest predicted on {} test dataset when wrong = {:.2%}".format(DATANAME, mean(high_wrong_unc)))

    print("")
    print("Mean probability on all not true label on {} test dataset = {:.2%}".format(DATANAME, mean(not_true_label_prob))) 
    print("Mean uncertainty on all not true label on {} test dataset = {:.2%}".format(DATANAME, mean(not_true_label_unc)))

    Uncertainty_output(NUM_CLASSES).scatterplot(correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, methodname, DATANAME)


def test_on_new_func(new_images_predictions, x_pred, methodname, y_pred = None, more_info=False):
    new_var_pred = []
    new_acc_pred = []
    new_var_not_pred = []
    new_acc_not_pred = []
    all_accuracies = []
    all_uncertainties = []

    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in new_images_predictions])
        predicted_ind = p_0.mean(axis=0).argmax()
        # print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
            # print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))
            all_accuracies.append(prob)
            all_uncertainties.append(var)

            if l == predicted_ind:
                new_var_pred.append(var)
                new_acc_pred.append(prob)
            else:
                new_var_not_pred.append(var)
                new_acc_not_pred.append(prob)

    print("")
    print("Mean probability on highest predicted class of {} data = {:.2%}".format(NEW_DATA, mean(new_acc_pred)))
    print("Mean uncertainty on highest predicted class of {} data = {:.2%}".format(NEW_DATA, mean(new_var_pred)))
    print("Mean probability on not predicted classes of {} data = {:.2%}".format(NEW_DATA, mean(new_acc_not_pred)))
    print("Mean uncertainty on not predicted classes of {} data = {:.2%}".format(NEW_DATA, mean(new_var_not_pred)))
    
    scatterplot(all_accuracies, all_uncertainties, methodname, 'new_all')

    if LABELS_AVAILABLE:
        pred = np.array(new_images_predictions).mean(axis=0).argmax(axis=1)
        acc = accuracy_score(y_pred.argmax(axis=1), pred)
        print("{} combined accuracy: {:.1%}".format(methodname, acc))

        confusion = tf.math.confusion_matrix(labels=y_pred.argmax(axis=1), predictions=pred,
                                        num_classes=NUM_CLASSES)
        print(confusion)

        true_labels = [np.argmax(i) for i in y_pred]
        wrong = 0
        correct = 0
        # Only when correctly predict, info of true class
        correct_unc = []
        correct_prob = []
        # Only when wrongly predicted, info of highest wrong pred
        high_wrong_unc = []
        high_wrong_prob = []
        # Only when wrongly predicted, info of true class
        true_wrong_unc = []
        true_wrong_prob = []        
        # Info of all incorrect classes
        all_wrong_unc = []
        all_wrong_prob = []
        # Info of all not true label classes
        not_true_label_unc = []
        not_true_label_prob = []
        # Info of all classes
        all_probabilities = []
        all_uncertainties = []

        for ind in range(len(y_pred)):
            all_predictions_single = np.array([p[ind] for p in new_images_predictions])
            # print("posterior mean: {}".format(all_predictions_single.mean(axis=0).argmax()))
            
            # probability + variance
            true_label = true_labels[ind]
            highest_pred_ind = all_predictions_single.mean(axis=0).argmax()

            for l, (prob, var) in enumerate(zip(all_predictions_single.mean(axis=0), all_predictions_single.std(axis=0))):
                all_probabilities.append(prob)
                all_uncertainties.append(var)

                if l == true_label:
                    if highest_pred_ind == true_label:
                        correct += 1
                        correct_unc.append(var)
                        correct_prob.append(prob)

                    else:
                        wrong += 1
                        true_wrong_unc.append(var)
                        true_wrong_prob.append(prob)

                        all_wrong_unc.append(var)
                        all_wrong_prob.append(prob)
                
                if l == highest_pred_ind:
                    high_wrong_unc.append(var)
                    high_wrong_prob.append(prob)

                else:
                    all_wrong_unc.append(var)
                    all_wrong_prob.append(prob)

                    not_true_label_unc.append(var)
                    not_true_label_prob.append(prob)  



        print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(correct+wrong))*100))
        print("")
        print("Mean probability on true label of {} test dataset when correctly predicted = {:.2%}".format(NEW_DATA, mean(correct_prob)))
        print("Mean uncertainty on true label of {} test dataset when correctly predicted = {:.2%}".format(NEW_DATA, mean(correct_unc)))
        print("Mean probability on true label of {} test dataset when wrongly predicted = {:.2%}".format(NEW_DATA, mean(true_wrong_prob))) 
        print("Mean uncertainty on true label of {} test dataset when wrongly predicted = {:.2%}".format(NEW_DATA, mean(true_wrong_unc)))    

        print("")
        print("Mean probability on highest predicted on {} test dataset when wrong = {:.2%}".format(NEW_DATA, mean(high_wrong_prob))) 
        print("Mean uncertainty on highest predicted on {} test dataset when wrong = {:.2%}".format(NEW_DATA, mean(high_wrong_unc)))

        print("")
        print("Mean probability on all not true label on {} test dataset = {:.2%}".format(NEW_DATA, mean(not_true_label_prob))) 
        print("Mean uncertainty on all not true label on {} test dataset = {:.2%}".format(NEW_DATA, mean(not_true_label_unc)))


        Uncertainty_output(NUM_CLASSES).scatterplot(correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, methodname, NEW_DATA)


    if more_info:
        for i in range(len(x_pred)):
            p_0 = np.array([p[i] for p in new_images_predictions])
            print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
            # probability + variance
            for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))


def scatterplot(accuracies, uncertainties, methodname, own_or_new):
    # os.chdir(HOME_DIR + os.path.sep + methodname)

    plt.scatter(accuracies, uncertainties)
    plt.xlabel('probability')
    plt.ylabel('uncertainty')
    plt.title('Scatterplot for {} on {}'.format(methodname, own_or_new))
    plt.savefig('{}_scatter_{}.png'.format(methodname, own_or_new))
    plt.clf()

    # os.chdir(HOME_DIR)


def save_array(filename, nparray):
    np.save(filename, nparray)


@profile
def MCDO(q, METHODNAME):

    MCDO_PREDICTIONS = 250

    if DATANAME == 'MES':
        MCDO_BATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-05-11_12-31_imagenet_64B_60.3%A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-05-11_12-09_imagenet_32B_71.9%A'
        MODEL_NAME = 'MCDO_model.h5'
        DATA_PATH = None

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-03-22_10-56_imagenet_16B_61.1%A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    def mcdo_predict(pre_trained_model, x_pred):
        mcdo_predictions = []
        datagen = ImageDataGenerator(rescale=1./255)
        pred_generator = datagen.flow(x_pred, batch_size=MCDO_BATCH_SIZE, shuffle=False)
        # progress_bar = tf.keras.utils.Progbar(target=MCDO_PREDICTIONS, interval=5)
        for i in range(MCDO_PREDICTIONS):
            # progress_bar.update(i)
            y_p = pre_trained_model.predict(pred_generator)
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

        os.chdir(HOME_DIR)
        os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
        print(os.getcwd())

        # Reload the model from the 2 files we saved
        with open('MCDO_model_config.json') as json_file:
            json_config = json_file.read()
        pre_trained_model = tf.keras.models.model_from_json(json_config)
        pre_trained_model.load_weights('MCDO_weights.h5')

        # pre_trained_model.summary()

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            mcdo_predictions = mcdo_predict(pre_trained_model, x_test)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_MCDO_' + DATANAME, mcdo_predictions)
            os.chdir(old_dir)
            test_on_own_func(METHODNAME, mcdo_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            mcdo_new_images_predictions = mcdo_predict(pre_trained_model, x_pred)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_MCDO_' + NEW_DATA, mcdo_new_images_predictions)
            os.chdir(old_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(mcdo_new_images_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(mcdo_new_images_predictions, x_pred, METHODNAME)
        
        else:
            mcdo_new_images_predictions = mcdo_predict(pre_trained_model, x_pred)
            test_on_new_func(mcdo_new_images_predictions, x_pred, METHODNAME, more_info = True)

        os.chdir(HOME_DIR)

    method_main()


@profile
def MCBN(q, METHODNAME):
    MCBN_PREDICTIONS = 250
    LEARN_RATE = 1

    if DATANAME == 'MES':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-05-11_14-13_imagenet_64B_59.4%A'
        MODEL_NAME = 'MCBN_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-05-11_12-09_imagenet_32B_82.2%A'
        MODEL_NAME = 'MCBN_model.h5'
        DATA_PATH = None

    if DATANAME == 'POLAR':
        MINIBATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODNAME
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
        datagen = ImageDataGenerator(rescale=1./255)
        pred_generator = datagen.flow(x_pred, batch_size=64, shuffle=False)
        # progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)

        org_model = pre_trained_model
        for i in range(MCBN_PREDICTIONS):
            # progress_bar.update(i)
            # Create new random minibatch from train data
            x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
            x_minibatch = np.asarray(x_minibatch)
            
            minibatch_generator = datagen.flow(x_minibatch,
                                        y_minibatch,
                                        batch_size=MINIBATCH_SIZE)

            # Fit the BN layers with the new minibatch, leave all other weights the same
            MCBN_model = org_model
            MCBN_model.fit(minibatch_generator,
                                epochs=1,
                                verbose=0)

            y_p = MCBN_model.predict(pred_generator) #Predict for bn look at (sigma and mu only one to chance, not the others)
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

        os.chdir(HOME_DIR)
        os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
        print(os.getcwd())

        # Reload the model from the 2 files we saved
        with open('MCBN_model_config.json') as json_file:
            json_config = json_file.read()
        pre_trained_model = tf.keras.models.model_from_json(json_config)
        pre_trained_model.load_weights('MCBN_weights.h5')

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        # Set onoly batch normalization layers to trainable
        for layer in pre_trained_model.layers:
            if re.search('batch_normalization.*', layer.name):
                layer.trainable = True
            else:
                layer.trainable = False
            # print(layer.name, layer.trainable)

        adam = optimizers.Adam(lr=LEARN_RATE)
        sgd = optimizers.SGD(lr=LEARN_RATE, momentum=0.9)

        OPTIMZ = sgd
        
        pre_trained_model.compile(
            optimizer=OPTIMZ,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # pre_trained_model.summary()  
   
        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            mcbn_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_test)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_MCBN_' + DATANAME, mcbn_predictions)
            os.chdir(old_dir)
            test_on_own_func(METHODNAME, mcbn_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            mcbn_new_images_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_pred)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_MCBN_' + NEW_DATA, mcbn_new_images_predictions)
            os.chdir(old_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(mcbn_new_images_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(mcbn_new_images_predictions, x_pred, METHODNAME)
        
        else:
            mcbn_new_images_predictions = mcbn_predict(pre_trained_model, x_train, y_train, x_pred)
            test_on_new_func(mcbn_new_images_predictions, x_pred, METHODNAME, more_info = True)

        os.chdir(HOME_DIR)

    method_main()


@profile
def Ensemble(q, METHODNAME):
    if DATANAME == 'MES':
        # Hyperparameters Messidor
        N_FOLDERS = 1
        N_ENSEMBLE_MEMBERS = [40]
        MODEL_VERSION = ['2020-05-11_12-19-17']
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5


    if DATANAME == 'CIFAR10':
        # Hyperparameters Cifar
        N_FOLDERS = 1
        # N_ENSEMBLE_MEMBERS = [20, 20]
        N_ENSEMBLE_MEMBERS = [40]
        # MODEL_VERSION = ['CIF_ImageNet_32B_20EN', 'CIF_ImageNet_32B_20EN_2']
        MODEL_VERSION = ['2020-05-11_12-01-39']
        DATA_PATH = None


    if DATANAME == 'POLAR':
        # Hyperparameters Polar
        N_FOLDERS = 1
        N_ENSEMBLE_MEMBERS = [40]
        MODEL_VERSION = ['2020-03-30_09-58_imagenet_16B_61.1%A']
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    MODEL_TO_USE = os.path.sep + METHODNAME


    def ensemble_predict(x_pred):
        mc_predictions = []
        datagen = ImageDataGenerator(rescale=1./255)
        pred_generator = datagen.flow(x_pred, batch_size=64, shuffle=False)
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
                y_p = pre_trained_model.predict(pred_generator)
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

        os.chdir(HOME_DIR)
        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            ensemble_predictions = ensemble_predict(x_test)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_Ensemble_' + DATANAME, ensemble_predictions)
            os.chdir(old_dir)
            test_on_own_func(METHODNAME, ensemble_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            ensemble_new_images_predictions = ensemble_predict(x_pred)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_Ensemble_' + NEW_DATA, ensemble_new_images_predictions)
            os.chdir(old_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(ensemble_new_images_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(ensemble_new_images_predictions, x_pred, METHODNAME)
        
        else:
            ensemble_only_new_images_predictions = ensemble_predict(x_pred)
            test_on_new_func(ensemble_new_images_predictions, x_pred, METHODNAME, more_info = True)

        os.chdir(HOME_DIR)

    method_main()
    # TODO: save testimage with in title predicted class, acc and prob


@profile
def VarianceOutput(q, METHODNAME):
    # Hyperparameters
    MODEL_TO_USE = os.path.sep + METHODNAME

    if DATANAME == 'MES':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + '2020-05-11_14-08_imagenet_64B_58.0%A'
        MODEL_NAME = 'Error_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + '2020-05-11_13-33_imagenet_64B_71.5%A'
        MODEL_NAME = 'Error_model.h5'
        DATA_PATH = None

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_VERSION = os.path.sep + '2020-03-29_10-41_imagenet_16B_64.8%A'
        MODEL_NAME = 'variance_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


    def convert_to_var(prediction):
        #Convert var pred to uncertainty
        for ind, pred in enumerate(prediction):
            predictions = pred[:NUM_CLASSES]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = np.abs(pred[NUM_CLASSES:])

            for i in range(0, NUM_CLASSES):
                pred_var = uncertainties[i]
                if i == highest_pred_ind:
                    # Highest predicted class, so error as if true
                    true_error = pow((predictions[i] - 1), 2)
                else:
                    # Not highest predicted class, so error as if false
                    true_error = pow((predictions[i]), 2)
                uncertainties[i] = abs(true_error - pred_var)
            prediction[ind][NUM_CLASSES:] = uncertainties   
         
        return prediction


    def var_label(org_data_prediction, y_test, scatter = True):
        # score on the test images (if label avaialable)
        true_labels = [np.argmax(i) for i in y_test]
        wrong = 0
        correct = 0
        # Only when correctly predict, info of true class
        correct_unc = []
        correct_prob = []
        # Only when wrongly predicted, info of highest wrong pred
        high_wrong_unc = []
        high_wrong_prob = []
        # Only when wrongly predicted, info of true class
        true_wrong_unc = []
        true_wrong_prob = []        
        # Info of all incorrect classes
        all_wrong_unc = []
        all_wrong_prob = []
        # Info of all not true label classes
        not_true_label_unc = []
        not_true_label_prob = []
        # Info of all classes
        all_probabilities = []
        all_uncertainties = []

        for ind, pred in enumerate(org_data_prediction):
            true_label = true_labels[ind]
            predictions = pred[:NUM_CLASSES]
            highest_pred_ind = np.argmax(predictions)
            uncertainties = pred[NUM_CLASSES:]

            for i in range(0, NUM_CLASSES):
                all_probabilities.append(predictions[i])
                all_uncertainties.append(uncertainties[i])

                if i == true_label:
                    if highest_pred_ind == true_label:
                        correct += 1
                        correct_unc.append(uncertainties[i])
                        correct_prob.append(predictions[i])

                    else:
                        wrong += 1
                        true_wrong_unc.append(uncertainties[i])
                        true_wrong_prob.append(predictions[i])

                        high_wrong_unc.append(uncertainties[highest_pred_ind])
                        high_wrong_prob.append(predictions[highest_pred_ind])

                        all_wrong_unc.append(uncertainties[i])
                        all_wrong_prob.append(predictions[i])                    

                else:
                    all_wrong_unc.append(uncertainties[i])
                    all_wrong_prob.append(predictions[i])

                    not_true_label_unc.append(uncertainties[i])
                    not_true_label_prob.append(predictions[i])                    
                    

        acc = accuracy_score(y_test.argmax(axis=1), org_data_prediction[:, :NUM_CLASSES].argmax(axis=1))
        print("Accuracy on original test dataset: {:.1%}".format(acc))

        confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=org_data_prediction[:, :NUM_CLASSES].argmax(axis=1),
                                        num_classes=NUM_CLASSES)
        print(confusion)

        print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(correct+wrong))*100))
        print("")
        print("Mean probability on true label of original test dataset when correctly predicted = {:.2%}".format(mean(correct_prob)))
        print("Mean uncertainty on true label of original test dataset when correctly predicted = {:.2%}".format(mean(correct_unc)))
        print("Mean probability on true label of original test dataset when wrongly predicted = {:.2%}".format(mean(true_wrong_prob))) 
        print("Mean uncertainty on true label of original test dataset when wrongly predicted = {:.2%}".format(mean(true_wrong_unc)))    

        print("")
        print("Mean probability on highest predicted on original test dataset when wrong = {:.2%}".format(mean(high_wrong_prob))) 
        print("Mean uncertainty on highest predicted on original test dataset when wrong = {:.2%}".format(mean(high_wrong_unc)))

        print("")
        print("Mean probability on all not true label on original test dataset = {:.2%}".format(mean(not_true_label_prob))) 
        print("Mean uncertainty on all not true label on original test dataset = {:.2%}".format(mean(not_true_label_unc)))

        if scatter:
            scatterplot(all_probabilities, all_uncertainties, METHODNAME, 'own_all')

            # Two scatterplots, one for the wrong predictions, one for the right predictions
            scatterplot(correct_prob, correct_unc, METHODNAME, 'own_label_correct')
            scatterplot(all_wrong_prob, all_wrong_unc, METHODNAME, 'own_label_wrong')

        else:
            # Two scatterplots, one for the wrong predictions, one for the right predictions
            scatterplot(correct_prob, correct_unc, METHODNAME, 'new_label_correct')
            scatterplot(all_wrong_prob, all_wrong_unc, METHODNAME, 'new_label_wrong')            


    def var_no_label(new_images_predictions, more_info = False):
        new_var_pred = []
        new_acc_pred = []
        new_var_not_pred = []
        new_acc_not_pred = []
        all_accuracies = []
        all_uncertainties = []

        for ind, pred in enumerate(new_images_predictions):
            classif = pred[:NUM_CLASSES]
            classif_ind = np.argmax(classif)
            var = pred[NUM_CLASSES:]
            var_pred = var[classif_ind]
            new_var_pred.append(var_pred)
            new_acc_pred.append(classif[classif_ind])
            
            for i in range(0, NUM_CLASSES):
                all_accuracies.append(classif[i])
                all_uncertainties.append(var[i])

                if classif_ind != i:
                    new_var_not_pred.append(var[i])
                    new_acc_not_pred.append(classif[i])

        print("")
        print("Mean uncertainty on highest predicted class of new data = {:.2%}".format(mean(new_var_pred)))
        print("Mean uncertainty on not predicted classes of new data = {:.2%}".format(mean(new_var_not_pred)))
        print("Mean probability on highest predicted class of new data = {:.2%}".format(mean(new_acc_pred)))
        print("Mean probability on not highest predicted class of new data = {:.2%}".format(mean(new_acc_not_pred)))

        if more_info:
            for ind, pred in enumerate(new_images_predictions):
                classif = pred[:NUM_CLASSES]
                classif_ind = np.argmax(classif)
                var = abs(pred[NUM_CLASSES:])
                if_true_error = pow((classif[classif_ind] - 1), 2)
                var_pred = abs(if_true_error - var[classif_ind])
                print("Predicted class: {}, Uncertainty of prediction: {:.2%}".format(classif_ind, var_pred))

        scatterplot(all_accuracies, all_uncertainties, METHODNAME, 'new_data')


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

        os.chdir(HOME_DIR)
        os.chdir(ROOT_PATH + os.path.sep + ONE_HIGHER_PATH[1] + MODEL_TO_USE + os.path.sep + DATANAME + MODEL_VERSION + os.path.sep)
        print(os.getcwd())

        # Reload the model from the 2 files we saved
        with open('Error_model_config.json') as json_file:
            json_config = json_file.read()
        pre_trained_model = tf.keras.models.model_from_json(json_config)
        pre_trained_model.load_weights('Error_weights.h5')
        # pre_trained_model.summary()
        os.chdir(HOME_DIR)

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        datagen = ImageDataGenerator(rescale=1./255)
        test_generator = datagen.flow(x_test, batch_size=64, shuffle=False)
        if TEST_ON_OWN_AND_NEW_DATASET:
            pred_generator = datagen.flow(x_pred, batch_size=64, shuffle=False)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            variance_org_dataset = pre_trained_model.predict(test_generator)
            variance_org_dataset = convert_to_var(variance_org_dataset)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_Error_' + DATANAME, variance_org_dataset)
            os.chdir(old_dir)
            Uncertainty_output(NUM_CLASSES).results_if_label(variance_org_dataset, y_test, scatter=True, name = DATANAME)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            variance_new_images_predictions = pre_trained_model.predict(pred_generator)
            variance_new_images_predictions = convert_to_var(variance_new_images_predictions)
            old_dir = os.getcwd()
            os.chdir(HOME_DIR)
            save_array(DATANAME + '_Error_' + NEW_DATA, variance_new_images_predictions)
            os.chdir(old_dir)
            if LABELS_AVAILABLE:
                Uncertainty_output(NUM_CLASSES).results_if_label(variance_new_images_predictions, y_pred, scatter=True, name = NEW_DATA)
        
        else:
            variance_new_images_predictions = pre_trained_model.predict(pred_generator)
            variance_new_images_predictions = convert_to_var(variance_new_images_predictions)
            Uncertainty_output(NUM_CLASSES).results_if_no_label(variance_new_images_predictions, more_info = True)

        os.chdir(HOME_DIR)

    method_main()


@profile
def main():
    for method_ind, METHODNAME in enumerate(METHODNAMES):
        if METHODNAME == 'MCDO':
            print('')
            print("START MCDO")
            # MCDO(METHODNAME)
            q = Queue()
            p = Process(target=MCDO, args=(q, METHODNAME))
            p.start()
            p.join() # this blocks until the process terminates
        if METHODNAME == 'MCBN':
            print('')
            print("START MCBN")
            # MCBN(METHODNAME)
            q = Queue()
            p = Process(target=MCBN, args=(q, METHODNAME))
            p.start()
            p.join() # this blocks until the process terminates
        if METHODNAME == 'Ensemble':
            print('')
            print("START Ensemble")
            # Ensemble(METHODNAME)
            q = Queue()
            p = Process(target=Ensemble, args=(q, METHODNAME))
            p.start()
            p.join() # this blocks until the process terminates
        if METHODNAME == 'VarianceOutput':
            print('')
            print("START VarianceOutput")
            q = Queue()
            # VarianceOutput(METHODNAME)
            p = Process(target=VarianceOutput, args=(q, METHODNAME))
            p.start()
            p.join() # this blocks until the process terminates

if __name__ == "__main__":
    main()
    print('')
    print("#################################")
    print_prof_data()
    print("#################################")



# TODO for varianceoutput train variance on different portion of data (different then for first loss)
# Truckje: laatste laag VGG16 opblazen () als output verandert betekent dat data niet gebalanceerd is
# Wanneer niet veranderd, dan overfitting
# x = Dense(4096, activation='relu', name='fc1')(x) -> 4096*2 of meer
# wanneer niet alles zelfde input resolutie, niet rescale (want kan je data verliezen en aspect ratio)
# maar, random noise rond correcte crop tot goede size, netwerk leert random noise te negeren


# TODO difference between higher uncertainty for wrong and just lower acc for wrong

# Connect threshold for what uncertainty means to accuracy of model on test set
#   (for example, given 85% accuracy on testset every classification with uncertainty above 5% is not to be trusted)

# Variacne output: nice thing that it can also confirm that it is certain of low probs for classes it doesnt think are it

# Check how many wrong predictions we can filter out 




