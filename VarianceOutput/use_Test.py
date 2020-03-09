''' Returns the prediction and uncertainty of images in /test_images of a pretrained (dropout)
 network of choice '''

import os
import datetime
import glob
import csv
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# input image dimensions
img_rows, img_cols = 28, 28

# Hyperparameters
NUM_CLASSES = 10
MODEL_TO_USE = os.path.sep + 'VarianceOutput'
MODEL_VERSION = '/2020-03-09_13-00-36'


DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0] 


def main():
    ''' Main function '''
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES*2)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES*2)


    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + MODEL_TO_USE + MODEL_VERSION + os.path.sep)

    # Reload the model from the 2 files we saved
    with open('variance_model_config.json') as json_file:
        json_config = json_file.read()
    pre_trained_model = tf.keras.models.model_from_json(json_config)
    pre_trained_model.load_weights('path_to_my_weights.h5')
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

    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = np.abs(pred[NUM_CLASSES:])
        var_wrong = var[classif_ind]
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

    # print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (correct/(correct + wrong))*100))
    # print("Supercorrect: {}, superwrong: {}, accuracy: {}%".format(supercorrect, notsupercorrect, (supercorrect/(supercorrect + notsupercorrect))*100))
    # print("match: {}, notmatch: {}, accuracy: {}%".format(match, notmatch, (match/(match + notmatch))*100))


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
    fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    os.chdir(fig_dir)

    print("")
    var_list = [[] for _ in range(NUM_CLASSES)]
    bins = np.arange(0, 0.2, 0.001).tolist()
    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        var_list[true_label].append(np.abs(pred[NUM_CLASSES:]))

    for lab in range(0, NUM_CLASSES):
        label_dir =  os.path.join(fig_dir, str(lab))
        os.makedirs(label_dir)
        os.chdir(label_dir)
        hist_list = [[] for _ in range(NUM_CLASSES)]
        for vars in var_list[lab]:
            for label, variance in enumerate(vars):
                hist_list[label].append(variance)
        for x in range(0, NUM_CLASSES):
            plt.hist(hist_list[x], bins = bins)
            plt.savefig('hist of vars of label_' + str(x) + '.png')
            plt.clf()
        os.chdir(fig_dir)


if __name__ == "__main__":
    main()
