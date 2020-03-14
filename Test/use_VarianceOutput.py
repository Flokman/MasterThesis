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
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Hyperparameters
NUM_CLASSES = 5
MODEL_TO_USE = os.path.sep + 'VarianceOutput'
MODEL_VERSION = '/2020-03-10_12-04-07 MES'

TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'
LABELS_AVAILABLE = False
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to

DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0] 

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
            x_pred = img_ar.reshape((1,) + img_ar.shape)

        else:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            img_ar = img_ar.reshape((1,) + img_ar.shape)
            x_pred = np.vstack((x_pred, img_ar))

    x_pred = x_pred.astype('float32')
    x_pred /= 255
    print('x_pred shape:', x_pred.shape)
    print(x_pred.shape[0], 'x_pred samples')

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


def main():
    ''' Main function '''

    if LABELS_AVAILABLE:
        (x_pred, y_pred) = load_new_images()
    else:
        x_pred = load_new_images()

    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + MODEL_TO_USE + MODEL_VERSION + os.path.sep)

    # Reload the model from the 2 files we saved
    with open('variance_model_config.json') as json_file:
        json_config = json_file.read()
    pre_trained_model = tf.keras.models.model_from_json(json_config)
    pre_trained_model.load_weights('path_to_my_weights.h5')
    # pre_trained_model.summary()
    os.chdir(old_dir)

    variance_predictions = pre_trained_model.predict(x_pred)

    if LABELS_AVAILABLE:
        # score on the test images (if label avaialable)
        true_labels = [np.argmax(i) for i in y_pred]
        wrong = 0
        correct = 0

        for ind, pred in enumerate(variance_predictions):
            true_label = true_labels[ind]
            classif = pred[:NUM_CLASSES]
            classif_ind = np.argmax(classif)
            var = pred[NUM_CLASSES:]
            var_wrong = var[classif_ind]
            var_correct = var[true_label]

            if classif_ind != true_label:
                wrong += 1
                print("Pred: {}, true: {}".format(classif_ind, true_label))
                print("Var_pred: {}, var_true: {}".format(var_wrong, var_correct))
            else:
                correct += 1
        
        print("Correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, 100- (wrong/correct)*100))

    else:
        for ind, pred in enumerate(variance_predictions):
            classif = pred[:NUM_CLASSES]
            classif_ind = np.argmax(classif)
            var = abs(pred[NUM_CLASSES:])
            if_true_error = pow((classif[classif_ind] - 1), 2)
            var_pred = abs(if_true_error - var[classif_ind])
            print("Predicted class: {}, Uncertainty of prediction: {:.2%}".format(classif_ind, var_pred))

            for i in range(len(x_pred)):
                raw_var = var[i]
                if classif[i] >= 0.5:
                    if_true_error = pow((classif[i] - 1), 2)
                else:
                    if_true_error = pow((classif[i]), 2)
                difference = abs(if_true_error - raw_var)
                print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, classif[i], difference))



if __name__ == "__main__":
    main()
