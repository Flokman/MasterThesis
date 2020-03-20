''' Returns the prediction and uncertainty of images in /test_images of a pretrained (dropout)
 network of choice '''

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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras import backend as K

# from customLoss import CategoricalVariance

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3
DATASET_NAME = 'CIFAR10'


TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = True
AUGMENTATION = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = True

# Hyperparameters
NUM_CLASSES = 10
MODEL_TO_USE = os.path.sep + 'VarianceOutput'
MODEL_VERSION = '/CIF_ImageNet_32B_95E_68A'


DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0] 
DATA_PATH = ROOT_PATH + os.path.sep + 'Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    # print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs_1 = {},
    #     epochts_2 = {}, test_img_idx = {},
    #     train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {},
    #     label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
    #     train_all_layers = {}, weights_to_use = {},
    #     es_patience_1 = {}, es_patience_2 = {}, train_val_split = {}""".format(
    #         DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS_1,
    #         EPOCHS_2, test_img_idx,
    #         TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION, label_count,
    #         LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
    #         TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
    #         ES_PATIENCE_1, ES_PATIENCE_2, TRAIN_VAL_SPLIT))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES*2)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES*2)

    return(x_train, y_train, x_test, y_test, test_img_idx)


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()


    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + MODEL_TO_USE + MODEL_VERSION + os.path.sep)

    # Reload the model from the 2 files we saved
    with open('variance_model_config.json') as json_file:
        json_config = json_file.read()
    pre_trained_model = tf.keras.models.model_from_json(json_config)
    pre_trained_model.load_weights('variance_weights.h5')
    # pre_trained_model.summary()
    os.chdir(old_dir)

    variance_predictions = pre_trained_model.predict(x_test)
    true_labels = [np.argmax(i) for i in y_test]
    wrong = 0
    correct = 0
    supercorrect = 0
    notsupercorrect = 0
    match = 0
    notmatch = 0
    varcorrect = 0
    varwrong = 0
    mean_var = []

    #Convert var pred to uncertainty
    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = np.abs(pred[NUM_CLASSES:])

        for i in range(0, NUM_CLASSES):
            raw_var = var[i]
            if_true_error = pow((classif[i] - y_test[ind][i]), 2)
            var[i] = abs(if_true_error - raw_var)
        variance_predictions[ind][NUM_CLASSES:] = var

    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = pred[NUM_CLASSES:]
        mean_var.append(np.mean(var))

        # for i in range(0, NUM_CLASSES):
        #     raw_var = var[i]
        #     if_true_error = pow((classif[i] - y_test[ind][i]), 2)
        #     var[i] = abs(if_true_error - raw_var)

        var_pred = var[classif_ind]
        var_correct = var[true_label]
        var_low = np.argmin(var)

        if classif_ind != true_label:
            wrong += 1
            # print("Pred: {}, true: {}".format(classif_ind, true_label))
            # print("Var_pred: {}, var_true: {}".format(var_wrong, var_correct))
        if classif_ind == true_label:
            correct += 1
        if classif_ind == true_label and classif_ind == var_low:
            supercorrect += 1
        if classif_ind != true_label and classif_ind != var_low:
            notsupercorrect += 1
        if classif_ind == var_low:
            match += 1
        if classif_ind != var_low:
            notmatch += 1
        if var_low == true_label:
            varcorrect +=1
        if var_low != true_label:
            varwrong  += 1
        
    
    total = len(variance_predictions)
    print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(total))*100))
    print("Varcorrect: {}, varwrong: {}, accuracy: {}%".format(varcorrect, varwrong, (varcorrect/(total))*100))
    print("Supercorrect: {}, superwrong: {}, accuracy: {}%".format(supercorrect, notsupercorrect, (supercorrect/(total))*100))
    print("match: {}, notmatch: {}, accuracy: {}%".format(match, notmatch, (match/(total))*100))
    print("Mean_var: {:.2%}, var_acc = {:.2%}".format(np.mean(mean_var), 1.0-(np.mean(mean_var))))

    # for i in range(0, 5):
    #     print("True label: {}".format(np.argmax(y_test[i])))
    #     pred = variance_predictions[i]
    #     # print(pred)
    #     classif = pred[:NUM_CLASSES]
    #     # classif = [(float(i)+1)/2 for i in classif]
    #     classif_max = np.amax(classif)
    #     classif_ind = np.argmax(classif)
    #     print(classif)
    #     print("Predicted value: {}, predicted class: {}".format(classif_max, classif_ind))

    #     var = np.abs(pred[NUM_CLASSES:])
    #     print(var)
    #     var_min = np.amin(var)
    #     var_ind = np.argmin(var)
    #     print("Min uncertainty: {}, min index: {}".format(var_min, var_ind))


    #     print("")
    #     print("Value of predicted class: {}".format(var[classif_ind]))
    #     print("##############################################################")

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(),('use_CIF_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(fig_dir)
    os.chdir(fig_dir)

    print("")
    var_list = [[] for _ in range(NUM_CLASSES)]
    bins = np.arange(0, 1, 0.025).tolist()
    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        var_list[true_label].append(pred[NUM_CLASSES:])

    fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11)) = plt.subplots(3,4, figsize=(20,15))
    for lab in range(0, NUM_CLASSES):
        hist_list = [[] for _ in range(NUM_CLASSES)]
        for varians in var_list[lab]:
            for label, variance in enumerate(varians):
                hist_list[label].append(variance)
        for x in range(0, NUM_CLASSES):
            if x == lab:
                eval('ax' + str(lab)).hist(hist_list[x], color='red', bins=bins, fill=True, label=x, histtype='step', stacked=True)
            else:
                eval('ax' + str(lab)).hist(hist_list[x], label=x, bins=bins, fill=False, histtype='step', stacked=True)
        eval('ax' + str(lab)).legend()
        eval('ax' + str(lab)).set_title('Class : {}'.format(lab))

        os.chdir(fig_dir)
    
    for ax in fig.get_axes():
        ax.label_outer()
    fig.tight_layout()
    fig.savefig('hist of all vars.png')

if __name__ == "__main__":
    main()
