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
EPOCHS = 100
N_ENSEMBLE_MEMBERS = 40
AMOUNT_OF_PREDICTIONS = 50
TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = True
AUGMENTATION = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = True
TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = None
LEARN_RATE = 0.0001
ES_PATIENCE = 30


def categorical_variance(y_true, y_pred, from_logits=False):
    y_pred = K.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    
    if y_true.shape[1] is not None:
        print("##############################################################")
        print(y_true, y_pred)
        num_class = int(y_true.shape[1] / 2)

        y_true_cat = y_true[:, :num_class]
        y_pred_cat = y_pred[:, :num_class]

        y_true_var = K.square(y_pred_cat - y_true_cat)
        # y_pred_var = y_pred[:, num_class:]
        
        y_true = K.concatenate([y_true_cat, y_true_var])
        tf.print(y_true)
        tf.print(y_pred)
        
        total_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        # total_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
        print(total_loss)
        print("##############################################################")

        return total_loss
    else:
        cat_loss = K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)
        return cat_loss


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


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
        loss=categorical_variance,
        metrics=[mean_pred]
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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='auto', verbose=1, patience=ES_PATIENCE)


    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)



    variance_model.fit(train_generator,
                epochs=EPOCHS,
                verbose=2,
                validation_data=val_generator,
                callbacks=[tensorboard_callback, early_stopping])

    score = variance_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    variance_predictions = variance_model.predict(x_test)
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

        var = pred[NUM_CLASSES:]
        print(var)
        var_min = np.amin(var)
        var_ind = np.argmin(var)
        print("Min uncertainty: {}, min index: {}".format(var_min, var_ind))


        print("")
        print("Value of predicted class: {}".format(var[classif_ind]))
        print("##############################################################")


if __name__ == "__main__":
    main()
