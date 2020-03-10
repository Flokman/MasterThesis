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
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential

# from customLoss import CategoricalVariance

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# input image dimensions
img_rows, img_cols = 28, 28


BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS_1 = 12
ES_PATIENCE_1 = 10
EPOCHS_2 = 100
ES_PATIENCE_2 = 5

TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9

LEARN_RATE = 0.0001


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

    y_pred_cat_abs = K.abs(y_pred_cat)
    y_true_var = K.square(y_pred_cat_abs - y_true_cat)
    y_pred_var = y_pred[:, NUM_CLASSES:]
    var_loss = K.mean(K.square(y_pred_var - y_true_var), axis=-1)
    total_loss = cat_loss + var_loss

    return total_loss


    # # mask = K.greater_equal(LOSS_SWITCH_COUNTER, tf.constant(BATCH_SIZE))
    # y_true_var = K.square(y_pred_cat - y_true_cat)
    # y_pred_var = y_pred[:, NUM_CLASSES:]
    # var_loss = K.mean(K.square(y_pred_var - y_true_var), axis=-1)
    # total_loss = cat_loss + var_loss
    # # loss = K.switch(mask, total_loss, cat_loss)

    # return loss
    
    # if y_true.shape[1] is not None:
    #     print("##############################################################")
    #     print(y_true, y_pred)
    #     num_class = int(y_true.shape[1] / 2)

    #     y_true_cat = y_true[:, :num_class]
    #     y_pred_cat = y_pred[:, :num_class]
    #     cat_loss = K.categorical_crossentropy(y_true_cat, y_pred_cat, from_logits=from_logits)

    #     y_true_var = K.square(y_pred_cat - y_true_cat)
    #     y_pred_var = y_pred[:, num_class:]
    #     var_loss = K.mean(K.square(y_pred_var - y_true_var), axis=-1)
    #     total_loss = cat_loss + var_loss

    #     return total_loss


    #     # if K.less_equal(cat_loss, tf.constant(1)):
    #     #     y_true_var = K.square(y_pred_cat - y_true_cat)
    #     #     y_pred_var = y_pred[:, num_class:]
    #     #     var_loss = K.mean(K.square(y_pred_var - y_true_var), axis=-1)
    #     #     total_loss = cat_loss + var_loss

    #     #     return total_loss
    #     # else:
    #     #     return cat_loss

    #     # y_true = K.concatenate([y_true_cat, y_true_var])
    #     # tf.print(y_true)
    #     # tf.print(y_pred)
        
    #     # total_loss = K.mean(K.square(y_pred - y_true), axis=-1)
    #     # total_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
    #     # print(total_loss)
        
        

    #     print("##############################################################")

        
    # else:
    #     cat_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
    #     return cat_loss


def mean_pred(y_true, y_pred):
    return K.mean(y_true - y_pred)


def custom_act(x):
    return tf.clip_by_value(x, 0, 100)


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


    inputs = layers.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape)(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    last = Dropout(0.5)(x)
    classification = Dense(NUM_CLASSES, activation='softmax')(last)
    variance = Dense(NUM_CLASSES, activation='linear')(last)

    out = concatenate([classification, variance])

    variance_model = Model(inputs=inputs, outputs=out)

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
    fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_1)


    datagen = ImageDataGenerator(rescale=1./255)
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
    variance_model.save_weights('path_to_my_weights.h5')

    # score = variance_model.predict(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
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

    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = np.abs(pred[NUM_CLASSES:])

        for i in range(0, NUM_CLASSES):
            raw_var = var[i]
            if_true_error = pow((classif[i] - 1), 2)
            var[i] = abs(if_true_error - raw_var)

        var_pred = var[classif_ind]
        var_correct = var[true_label]
        var_low = np.argmin(var)



        if classif_ind != true_label:
            wrong += 1
            # print("Pred: {}, true: {}".format(classif_ind, true_label))
            # print("Var_pred: {}, var_true: {}".format(var_pred, var_correct))
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


    for i in range(0, 5):
        print("True label: {}".format(np.argmax(y_test[i])))
        pred = variance_predictions[i]
        # print(pred)
        classif = pred[:NUM_CLASSES]
        # classif = [(float(i)+1)/2 for i in classif]
        classif_max = np.amax(classif)
        classif_ind = np.argmax(classif)
        print(classif)
        print("Predicted value: {}, predicted class: {}".format(classif_max, classif_ind))

        var = np.abs(pred[NUM_CLASSES:])
        print(var)
        var_min = np.amin(var)
        var_ind = np.argmin(var)
        print("Min uncertainty: {}, min index: {}".format(var_min, var_ind))


        print("")
        print("Value of predicted class: {}".format(var[classif_ind]))
        print("##############################################################")


if __name__ == "__main__":
    main()
