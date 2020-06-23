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
DATANAME = 'CIFAR10'
NEW_DATA = 'MNIST' # or 'local' for local folder
METHODNAMES = ['MCDO', 'MCBN', 'Ensemble', 'VarianceOutput']
# METHODNAMES = ['MCDO', 'MCBN', 'Ensemble']

TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.875

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
            x_train, y_train, x_test, y_test = np.array(f['x_train']), np.array(f['y_train']), np.array(f['x_test']), np.array(f['y_test'])
        label_count = [0] * NUM_CLASSES
        for lab in y_train:
            label_count[lab] += 1

        if to_shuffle:
            (x_load, y_load) = shuffle_data(x_load, y_load)

        # Divide the data into a train and test set
        x_test, x_val = np.split(x_test, [int(TRAIN_VAL_SPLIT*len(x_test))])
        y_test, y_val = np.split(y_test, [int(TRAIN_VAL_SPLIT*len(y_test))])

        return (x_train, y_train), (x_val, y_val), (x_test, y_test), label_count


def load_hdf5_dataset(DATA_PATH, new_data=False):
    ''' Load a dataset, split and put in right format'''

    # Split the data between train and test sets
    if not new_data:
        if DATANAME == 'POLAR':
            (x_train, y_train), (x_test, y_test), train_label_count, test_label_count = load_data(DATA_PATH, TO_SHUFFLE)

        if DATANAME == 'MES':
            (x_train, y_train), (x_val, y_val), (x_test, y_test), label_count = load_data(DATA_PATH, TO_SHUFFLE)
            x_val = np.asarray(x_val)
            y_val = np.asarray(y_val)

        if DATANAME == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_test, x_val = np.split(x_test, [int(TRAIN_VAL_SPLIT*len(x_test))])
            y_test, y_val = np.split(y_test, [int(TRAIN_VAL_SPLIT*len(y_test))])

    else:
        if NEW_DATA == 'MNIST':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)
            print("MNIST data returning")
            return((x_train, y_train), (x_test, y_test))

    
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


def load_new_images(img_height = IMG_HEIGHT):
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
                img = img.resize((img_height, img_height))
                img_ar = np.array(img)
                img_ar = np.stack((img_ar, img_ar, img_ar), axis=2)
                x_pred = img_ar.reshape((1,) + img_ar.shape)
            else:    
                img = Image.fromarray(img_ar)
                img = img.resize((img_height, img_height))
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

    Uncertainty_output(NUM_CLASSES).scatterplot(correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, methodname, DATANAME, ylabel_name = 'STD')
    

def test_on_own_funcV2(methodname, predictions, y_test):
    # uncertainties = gaussian_unc(predictions)
    # old_dir = os.getcwd()
    # os.chdir(HOME_DIR)
    # save_array(DATANAME + '_CDF_' + methodname, uncertainties)
    # os.chdir(old_dir)

    old_dir = os.getcwd()
    os.chdir(HOME_DIR)
    uncertainties = np.load(DATANAME + '_CDF_' + methodname + '.npy')
    os.chdir(old_dir)


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
        all_uncertanties_single = np.array([p[ind] for p in uncertainties])
        # print("posterior mean: {}".format(all_predictions_single.mean(axis=0).argmax()))
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = all_predictions_single.mean(axis=0).argmax()

            
        for l, (prob, var) in enumerate(zip(all_predictions_single.mean(axis=0), all_uncertanties_single.mean(axis=0))):
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

    Uncertainty_output(NUM_CLASSES).scatterplot(correct_prob, correct_unc, high_wrong_prob, high_wrong_unc, methodname, DATANAME, ylabel_name = 'PDF')


def MSE_STD(methodname, predictions, y_test):
    # mean_predictions_all = np.array(predictions).mean(axis=0).argmax(axis=1)
    # acc = accuracy_score(y_test.argmax(axis=1), mean_predictions_all)
    # print("{} combined accuracy: {:.1%}".format(methodname, acc))

    # confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mean_predictions_all,
    #                                 num_classes=NUM_CLASSES)
    # print(confusion)

    true_labels = [np.argmax(i) for i in y_test]
    
    # Squared errors per example per amount of predictions per example
    SEs = np.empty([len(y_test), predictions.shape[0]])
    # Variance for specific predicted class over all predictions for that class per example per amount of predictions per example
    VARs = np.empty([len(y_test)])

    for ind in range(len(y_test)):
        all_predictions_single = np.array([p[ind] for p in predictions])
        all_uncertanties_single = all_predictions_single.std(axis=0)
        
        mean_pred_single_examp = all_predictions_single.mean(axis=0)
        # mean_uncert_single_examp = all_uncertanties_single.mean(axis=0)
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = mean_pred_single_examp.argmax()
        # print(highest_pred_ind)
        # print(all_uncertanties_single)
        mean_uncert_highest_single_examp = all_uncertanties_single[highest_pred_ind]
        VARs[ind] = mean_uncert_highest_single_examp
        
        # print("all var single shape: ", mean_uncert_single_examp.shape)
        for index, single_pred in enumerate(all_predictions_single):
            # For this one single prediction, what is the index of highest class
            high_single_pred_ind = np.argmax(single_pred)

            # Calculate Squared Error
            if high_single_pred_ind == true_label:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind] - 1), 2)
            else:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind]), 2)
            
            # Accompining variance calculated over all predictions for that specific class
            # VARs[ind, index] = mean_uncert_single_examp[high_single_pred_ind]
        
    MSE = SEs.mean(axis=1)
    UNC = VARs
    print("MSE shape: ", MSE.shape)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    print("creating MSE scatterplot")
    plt.clf()
    plt.style.use("ggplot")
    plt.scatter(UNC, MSE, c='b')
    plt.legend()
    plt.ylabel('Uncertainty (STD)')
    plt.xlabel('MSE')
    plt.title('Scatterplot for {} on {}'.format(methodname, DATANAME))
    plt.savefig('{}_MSE_STD_scatter_{}.png'.format(methodname, DATANAME))
    plt.clf()


def MSE_PDF(methodname, predictions, y_test):
    old_dir = os.getcwd()
    os.chdir(HOME_DIR)
    uncertainties = np.load(DATANAME + '_CDF_' + methodname + '.npy')
    os.chdir(old_dir)    

    true_labels = [np.argmax(i) for i in y_test]
    
    # Squared errors per example per amount of predictions per example
    SEs = np.empty([len(y_test), predictions.shape[0]])
    # Variance for specific predicted class over all predictions for that class per example per amount of predictions per example
    VARs = np.empty([len(y_test)])

    for ind in range(len(y_test)):
        all_predictions_single = np.array([p[ind] for p in predictions])
        all_uncertanties_single = np.array([p[ind] for p in uncertainties])
        
        mean_pred_single_examp = all_predictions_single.mean(axis=0)
        mean_uncert_single_examp = all_uncertanties_single.mean(axis=0)
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = mean_pred_single_examp.argmax()
        mean_uncert_highest_single_examp = mean_uncert_single_examp[highest_pred_ind]
        VARs[ind] = mean_uncert_highest_single_examp
        
        # print("all var single shape: ", mean_uncert_single_examp.shape)
        for index, single_pred in enumerate(all_predictions_single):
            # For this one single prediction, what is the index of highest class
            high_single_pred_ind = np.argmax(single_pred)

            # Calculate Squared Error
            if high_single_pred_ind == true_label:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind] - 1), 2)
            else:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind]), 2)
            
            # Accompining variance calculated over all predictions for that specific class
            # VARs[ind, index] = mean_uncert_single_examp[high_single_pred_ind]
        
    MSE = SEs.mean(axis=1)
    UNC = VARs
    print("MSE shape: ", MSE.shape)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    print("creating MSE scatterplot")
    plt.clf()
    plt.style.use("ggplot")
    plt.scatter(UNC, MSE, c='b')
    plt.legend()
    plt.ylabel('Uncertainty (PDF)')
    plt.xlabel('MSE')
    plt.title('Scatterplot for {} on {}'.format(methodname, DATANAME))
    plt.savefig('{}_MSE_PDF_scatter_{}.png'.format(methodname, DATANAME))
    plt.clf()


def MSE_STDV2(methodname, predictions, y_test):
    # mean_predictions_all = np.array(predictions).mean(axis=0).argmax(axis=1)
    # acc = accuracy_score(y_test.argmax(axis=1), mean_predictions_all)
    # print("{} combined accuracy: {:.1%}".format(methodname, acc))

    # confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mean_predictions_all,
    #                                 num_classes=NUM_CLASSES)
    # print(confusion)

    true_labels = [np.argmax(i) for i in y_test]
    
    # Squared errors per example per amount of predictions per example
    SEs = np.empty([len(y_test), predictions.shape[0]])
    # Variance for specific predicted class over all predictions for that class per example per amount of predictions per example
    VARs = np.empty([len(y_test), predictions.shape[0]])

    for ind in range(len(y_test)):
        all_predictions_single = np.array([p[ind] for p in predictions])
        all_var_single = all_predictions_single.std(axis=0)
        # print("posterior mean: {}".format(all_predictions_single.mean(axis=0).argmax()))
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = all_predictions_single.mean(axis=0).argmax()

        
        # print("all var single shape: ", all_var_single.shape)
        for index, single_pred in enumerate(all_predictions_single):
            # For this one single prediction, what is the index of highest class
            high_single_pred_ind = np.argmax(single_pred)

            # Calculate Squared Error
            if high_single_pred_ind == true_label:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind] - 1), 2)
            else:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind]), 2)
            
            # Accompining variance calculated over all predictions for that specific class
            VARs[ind, index] = all_var_single[high_single_pred_ind]
        
    MSE = SEs.mean(axis=1)
    UNC = VARs.mean(axis=1)
    print("MSE shape: ", MSE.shape)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    print("creating MSE scatterplot")
    plt.clf()
    plt.style.use("ggplot")
    plt.scatter(UNC, MSE, c='b')
    plt.legend()
    plt.ylabel('Uncertainty (STD)')
    plt.xlabel('MSE')
    plt.title('Scatterplot for {} on {}'.format(methodname, DATANAME))
    plt.savefig('{}_MSE_STD_scatter_{}.png'.format(methodname, DATANAME))
    plt.clf()


def MSE_PDFV2(methodname, predictions, y_test):
    old_dir = os.getcwd()
    os.chdir(HOME_DIR)
    uncertainties = np.load(DATANAME + '_CDF_' + methodname + '.npy')
    os.chdir(old_dir)    

    true_labels = [np.argmax(i) for i in y_test]
    
    # Squared errors per example per amount of predictions per example
    SEs = np.empty([len(y_test), predictions.shape[0]])
    # Variance for specific predicted class over all predictions for that class per example per amount of predictions per example
    VARs = np.empty([len(y_test), predictions.shape[0]])

    for ind in range(len(y_test)):
        all_predictions_single = np.array([p[ind] for p in predictions])
        all_uncertanties_single = np.array([p[ind] for p in uncertainties])
        
        all_var_single = all_uncertanties_single.mean(axis=0)
        
        # probability + variance
        true_label = true_labels[ind]
        highest_pred_ind = all_predictions_single.mean(axis=0).argmax()

        
        # print("all var single shape: ", all_var_single.shape)
        for index, single_pred in enumerate(all_predictions_single):
            # For this one single prediction, what is the index of highest class
            high_single_pred_ind = np.argmax(single_pred)

            # Calculate Squared Error
            if high_single_pred_ind == true_label:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind] - 1), 2)
            else:
                SEs[ind, index] = pow((single_pred[high_single_pred_ind]), 2)
            
            # Accompining variance calculated over all predictions for that specific class
            VARs[ind, index] = all_var_single[high_single_pred_ind]
        
    MSE = SEs.mean(axis=1)
    UNC = VARs.mean(axis=1)
    print("MSE shape: ", MSE.shape)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    print("creating MSE scatterplot")
    plt.clf()
    plt.style.use("ggplot")
    plt.scatter(UNC, MSE, c='b')
    plt.legend()
    plt.ylabel('Uncertainty (PDF)')
    plt.xlabel('MSE')
    plt.title('Scatterplot for {} on {}'.format(methodname, DATANAME))
    plt.savefig('{}_MSE_PDF_scatter_{}.png'.format(methodname, DATANAME))
    plt.clf()


def gaussian_unc(predictions):
    from scipy.stats import norm
    # Predictions shape (N_models, N_examples_ N_classes)

    for exam in range(predictions.shape[1]):
        all_predictions_single = np.array([p[exam] for p in predictions]) # Shape (N_models, N_classes)
        std = all_predictions_single.std(axis=0)
        mean = all_predictions_single.mean(axis=0)
        gaussian_list = []
        ultrapdfs = []
        for i, s in enumerate(std):
            gaussian_list.append(norm(loc=mean[i], scale=s))
            ultrapdfs.append(gaussian_list[i].pdf(mean[i]))
        print("{}/{}".format(exam, predictions.shape[1]))


        for model_index, model_pred in enumerate(all_predictions_single):
            # print('########')
            for single_classs_index, single_class in enumerate(model_pred):
                pdf = gaussian_list[single_classs_index].pdf(single_class)
                # ultrapdf = gaussian_list[single_classs_index].pdf(mean[single_classs_index])
                uncer = 1 - (pdf / ultrapdfs[single_classs_index])
                # print("")
                # print(mean[single_classs_index], std[single_classs_index])
                # print(single_class)
                # print('PDF: ',pdf)
                # print('ultra PDF: ', ultrapdf)
                # print(uncer)
                # print("")
                predictions[model_index, exam, single_classs_index] = uncer                

    return predictions


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
    
    Uncertainty_output(NUM_CLASSES).scatterplot(None, None, all_accuracies, all_uncertainties, methodname, NEW_DATA)

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


def test_on_new_funcV2(new_images_predictions, x_pred, methodname, y_pred = None, more_info=False):
    uncertainties = gaussian_unc(new_images_predictions)
    old_dir = os.getcwd()
    os.chdir(HOME_DIR)
    save_array(DATANAME + '_PDF_' + methodname + '_' + NEW_DATA, uncertainties)
    os.chdir(old_dir)

    # old_dir = os.getcwd()
    # os.chdir(HOME_DIR)
    # uncertainties = np.load(DATANAME + '_PDF_' + methodname + '_' + NEW_DATA + '.npy')
    # os.chdir(old_dir)

    new_var_pred = []
    new_acc_pred = []
    new_var_not_pred = []
    new_acc_not_pred = []
    all_accuracies = []
    all_uncertainties = []

    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in new_images_predictions])
        all_uncertanties_single = np.array([p[ind] for p in uncertainties])
        predicted_ind = p_0.mean(axis=0).argmax()
        # print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), all_uncertanties_single.mean(axis=0))):
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
    
    Uncertainty_output(NUM_CLASSES).scatterplot(None, None, new_acc_pred, new_var_pred, methodname, NEW_DATA)

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

    plt.scatter(accuracies, uncertainties, c='r')
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
        MODEL_VERSION = os.path.sep + 'MES_ImageNet_Retrain_32B_93E_52A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5
        mcdo_own_predictions = np.load('MES_MCDO_MES.npy')

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + 'CIFAR_ImageNet_retrain_32B_95E_86A'
        MODEL_NAME = 'MCDO_model.h5'
        DATA_PATH = None
        mcdo_own_predictions = np.load('CIFAR10_MCDO_CIFAR10.npy')

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-03-22_10-56_imagenet_16B_61.1%A'
        MODEL_NAME = 'MCDO_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


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


        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            test_on_own_func(METHODNAME, mcdo_own_predictions, y_test)
            MSE_STD(METHODNAME, mcdo_own_predictions, y_test)
            test_on_own_funcV2(METHODNAME, mcdo_own_predictions, y_test)
            MSE_PDF(METHODNAME, mcdo_own_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            os.chdir(HOME_DIR)
            mcdo_new_predictions = np.load('CIFAR10_MCDO_MNIST.npy')
            os.chdir(fig_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(mcdo_new_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(mcdo_new_predictions, x_pred, METHODNAME)

        os.chdir(HOME_DIR)

    method_main()


@profile
def MCBN(q, METHODNAME):
    MCBN_PREDICTIONS = 250
    LEARN_RATE = 1

    if DATANAME == 'MES':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + 'ImageNet_retrain_32B_144E_88A'
        MODEL_NAME = 'MCBN_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5
        MCBN_own_predictions = np.load('MES_MCBN_MES.npy')

    if DATANAME == 'CIFAR10':
        MINIBATCH_SIZE = 128
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-04-10_14-30_imagenet_32B_82.3%A'
        MODEL_NAME = 'MCBN_model.h5'
        DATA_PATH = None
        MCBN_own_predictions = np.load('CIFAR10_MCBN_CIFAR10.npy')

    if DATANAME == 'POLAR':
        MINIBATCH_SIZE = 16
        MODEL_TO_USE = os.path.sep + METHODNAME
        MODEL_VERSION = os.path.sep + '2020-03-22_10-50_imagenet_8B_51.9%A'
        MODEL_NAME = 'MCBN_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


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
       

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)
   
        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            test_on_own_func(METHODNAME, MCBN_own_predictions, y_test)
            MSE_STD(METHODNAME, MCBN_own_predictions, y_test)
            test_on_own_funcV2(METHODNAME, MCBN_own_predictions, y_test)
            MSE_PDF(METHODNAME, MCBN_own_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            os.chdir(HOME_DIR)
            MCBN_new_predictions = np.load('CIFAR10_MCBN_MNIST.npy')
            os.chdir(fig_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(MCBN_new_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(MCBN_new_predictions, x_pred, METHODNAME)
        
        os.chdir(HOME_DIR)

    method_main()


@profile
def Ensemble(q, METHODNAME):
    if DATANAME == 'MES':
        # Hyperparameters Messidor
        N_FOLDERS = 2
        N_ENSEMBLE_MEMBERS = [43, 27]
        MODEL_VERSION = ['/MES_32B_43EN', '/MES_ImageNet_32B_27EN']
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5
        Ensemble_own_predictions = np.load('MES_Ensemble_MES.npy')


    if DATANAME == 'CIFAR10':
        # Hyperparameters Cifar
        N_FOLDERS = 1
        # N_ENSEMBLE_MEMBERS = [20, 20]
        N_ENSEMBLE_MEMBERS = [40]
        # MODEL_VERSION = ['CIF_ImageNet_32B_20EN', 'CIF_ImageNet_32B_20EN_2']
        MODEL_VERSION = ['2020-03-19_16-19-18']
        DATA_PATH = None
        Ensemble_own_predictions = np.load('CIFAR10_Ensemble_CIFAR10.npy')


    if DATANAME == 'POLAR':
        # Hyperparameters Polar
        N_FOLDERS = 1
        N_ENSEMBLE_MEMBERS = [40]
        MODEL_VERSION = ['2020-03-30_09-58_imagenet_16B_61.1%A']
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


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
      

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            test_on_own_func(METHODNAME, Ensemble_own_predictions, y_test)
            MSE_STD(METHODNAME, Ensemble_own_predictions, y_test)
            test_on_own_funcV2(METHODNAME, Ensemble_own_predictions, y_test)
            MSE_PDF(METHODNAME, Ensemble_own_predictions, y_test)
        
        if TEST_ON_OWN_AND_NEW_DATASET:
            os.chdir(HOME_DIR)
            Ensemble_new_predictions = np.load('CIFAR10_Ensemble_MNIST.npy')
            os.chdir(fig_dir)
            if LABELS_AVAILABLE:
                test_on_new_func(Ensemble_new_predictions, x_pred, METHODNAME, y_pred = y_pred)
            else:
                test_on_new_func(Ensemble_new_predictions, x_pred, METHODNAME)
        
        os.chdir(HOME_DIR)

    method_main()


@profile
def VarianceOutput(q, METHODNAME):
    # Hyperparameters
    MODEL_TO_USE = os.path.sep + METHODNAME

    if DATANAME == 'MES':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + 'MES_ImageNet_Retrain_32B_90E_52A'
        MODEL_NAME = 'variance_model.h5'
        DATASET_LOCATION = os.path.sep + 'Datasets'
        DATASET_HDF5 = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = DATA_ROOT_PATH + DATASET_LOCATION + DATASET_HDF5
        Error_own_predictions = np.load('MES_Error_MES.npy')

    if DATANAME == 'CIFAR10':
        MCDO_BATCH_SIZE = 128
        MODEL_VERSION = os.path.sep + 'CIF_ImageNet_32B_95E_68A'
        MODEL_NAME = 'variance_model.h5'
        DATA_PATH = None
        Error_own_predictions = np.load('CIFAR10_Error_CIFAR10.npy')

    if DATANAME == 'POLAR':
        MCDO_BATCH_SIZE = 16
        MODEL_VERSION = os.path.sep + '2020-03-29_10-41_imagenet_16B_64.8%A'
        MODEL_NAME = 'variance_model.h5'
        DATASET_HDF5 = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'
        DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_HDF5


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
            # x_pred = load_new_images()           

        elif TEST_ON_NEW_DATASET:
            (x_train, y_train), (x_test, y_test) = load_hdf5_dataset(DATA_PATH)
            # x_pred = load_new_images()
       

        fig_dir = os.path.join(HOME_DIR + os.path.sep + METHODNAME, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fig_dir)
        os.chdir(fig_dir)

        if TEST_ON_OWN_DATASET or LABELS_AVAILABLE or TEST_ON_OWN_AND_NEW_DATASET:
            Error_own_predictions_CONV = Uncertainty_output(NUM_CLASSES).convert_output_to_uncertainty(Error_own_predictions)
            Uncertainty_output(NUM_CLASSES).results_if_label(Error_own_predictions_CONV, y_test, scatter=True, name = DATANAME)
            Uncertainty_output(NUM_CLASSES).MSE(Error_own_predictions_CONV, y_test, name = DATANAME)
        if TEST_ON_OWN_AND_NEW_DATASET:
            os.chdir(HOME_DIR)
            Error_new_predictions = np.load('CIFAR10_Error_MNIST.npy')
            Error_own_predictions_CONV = Uncertainty_output(NUM_CLASSES).convert_output_to_uncertainty(Error_new_predictions)
            os.chdir(fig_dir)
            if LABELS_AVAILABLE:
                Uncertainty_output(NUM_CLASSES).results_if_label(Error_own_predictions_CONV, y_pred, scatter=True, name = NEW_DATA)
            else:
                Uncertainty_output(NUM_CLASSES).results_if_no_label(Error_own_predictions_CONV, scatter = True, name = NEW_DATA)
        
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




