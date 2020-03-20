''' Trains a (pre trained) network with additional dropout layers for uncertainty estimation'''

#https://stackoverflow.com/questions/49646304/keras-optimizing-two-outputs-with-a-custom-loss
# https://stackoverflow.com/questions/46663013/what-is-y-true-and-y-pred-when-creating-a-custom-metric-in-keras
# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

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
DATASET_NAME = os.path.sep + 'CIFAR10'

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS_1 = 120
ES_PATIENCE_1 = 30
EPOCHS_2 = 300
ES_PATIENCE_2 = 50

TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9

TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.00001


# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + os.path.sep + 'Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs_1 = {},
        epochts_2 = {}, test_img_idx = {},
        train_test_split = {}
        learn rate = {},
        train_all_layers = {}, weights_to_use = {},
        es_patience_1 = {}, es_patience_2 = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS_1,
            EPOCHS_2, test_img_idx,
            TRAIN_TEST_SPLIT,
            LEARN_RATE,
            TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
            ES_PATIENCE_1, ES_PATIENCE_2, TRAIN_VAL_SPLIT))

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


def categorical_cross(y_true, y_pred, from_logits=False):
    y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    y_true_cat = y_true[:, :NUM_CLASSES]
    y_pred_cat = y_pred[:, :NUM_CLASSES]
    # cat_loss = K.mean(K.square(y_pred_cat - y_true_cat), axis=-1)
    cat_loss = K.categorical_crossentropy(y_true_cat, y_pred_cat, from_logits=from_logits)
    
    return cat_loss


def categorical_variance(y_true, y_pred, from_logits=False):
    y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    y_true_cat = y_true[:, :NUM_CLASSES]
    y_pred_cat = y_pred[:, :NUM_CLASSES]
    # cat_loss = K.mean(K.square(y_pred_cat - y_true_cat), axis=-1)
    cat_loss = K.categorical_crossentropy(y_true_cat, y_pred_cat, from_logits=from_logits)

    # TODO: test first abs of y_pred_cat
    # Is error only modelled after being right? Or also wrong?
    y_pred_cat_abs = K.abs(y_pred_cat)
    y_true_var = K.square(y_pred_cat_abs - y_true_cat)
    y_pred_var = y_pred[:, NUM_CLASSES:]
    var_loss = K.mean(K.square(y_pred_var - y_true_var), axis=-1)
    total_loss = cat_loss + var_loss

    return total_loss

def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    # VGG16 since it does not include batch normalization of dropout by itself
    variance_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
                       classes=NUM_CLASSES)

    # Stacking a new simple convolutional network on top of vgg16
    all_layers = [l for l in variance_model.layers]
    x = all_layers[0].output
    for i in range(1, len(all_layers)):
        x = all_layers[i](x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    last_layer = Dense(4096, activation='relu', name='fc2')(x)
    classification = Dense(NUM_CLASSES, activation='softmax')(last_layer)
    variance = Dense(NUM_CLASSES, activation='linear')(last_layer)

    out = concatenate([classification, variance])

    # Creating new model
    variance_model = Model(inputs=all_layers[0].input, outputs=out)

    variance_model.summary()

    adam = optimizers.Adam(lr=LEARN_RATE)
    # sgd = optimizers.SGD(lr=LEARN_RATE)

    variance_model.compile(
        optimizer=adam,
        loss=categorical_cross,
        metrics=['acc']
    )

    print("Start fitting")
    
    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), "CIFAR10" + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_1)


    datagen = ImageDataGenerator(rescale=1./255, dtype='ndarray')
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)



    variance_model.fit(train_generator,
                epochs=EPOCHS_1,
                verbose=2,
                validation_data=val_generator,
                callbacks=[tensorboard_callback, early_stopping_1])

    variance_model.compile(
        optimizer=adam,
        loss=categorical_variance,
        metrics=['acc']
    )

    early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_2)

    variance_model.fit(train_generator,
                       epochs=EPOCHS_2,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback, early_stopping_2])

    # Save JSON config to disk
    json_config = variance_model.to_json()
    with open('variance_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    variance_model.save_weights('variance_weights.h5')

    variance_predictions = variance_model.predict(x_test)
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


if __name__ == "__main__":
    main()