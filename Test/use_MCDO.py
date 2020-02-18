''' Returns the prediction and uncertainty of images in /test_images of a pretrained (dropout)
 network of choice '''

import os
import datetime
import glob
import csv
import tensorflow as tf
import numpy as np
from astroNN import MCDropout

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Hyperparameters
NUM_CLASSES = 5
MCDO_PREDICTIONS = 250
MCDO_BATCH_SIZE = 250
MODEL_TO_USE = os.path.sep + 'Dropout'
MODEL_VERSION = '/2020-02-11_13-48-17'
MODEL_NAME = 'mcdo_model.h5'

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


def indepth_predictions(x_pred, y_pred, mc_predictions):
    ''' Creates more indepth predictions (print on command line and create figures in folder) '''
    fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # Dir to store created figures
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

    if LABELS_AVAILABLE:
        (x_pred, y_pred) = load_new_images()
    else:
        x_pred = load_new_images()

    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + MODEL_TO_USE + MODEL_VERSION + os.path.sep)
    print(os.getcwd())
    pre_trained_model = load_model(MODEL_NAME, custom_objects={'MCDropout': MCDropout})
    pre_trained_model.summary()
    os.chdir(old_dir)

    mc_predictions = []

    progress_bar = tf.keras.utils.Progbar(target=MCDO_PREDICTIONS, interval=5)
    for i in range(MCDO_PREDICTIONS):
        progress_bar.update(i)
        y_p = pre_trained_model.predict(x_pred, batch_size=MCDO_BATCH_SIZE)
        mc_predictions.append(y_p)

    if LABELS_AVAILABLE:
        # score on the test images (if label avaialable)
        indepth_predictions(x_pred, y_pred, mc_predictions)
    else:
        for i in range(len(x_pred)):
            p_0 = np.array([p[i] for p in mc_predictions])
            print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
            # probability + variance
            for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
                print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))


if __name__ == "__main__":
    main()