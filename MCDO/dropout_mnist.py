from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers


import numpy as np
import pandas as pd
import os, datetime, time
import h5py
from random import seed, randint, shuffle
import multiprocessing
import cv2
import re


from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) # Dir to store created figures
os.makedirs(fig_dir)
log_dir = os.path.join(fig_dir,"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Dir to store Tensorboard data
os.makedirs(log_dir)

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


batch_size = 128
num_classes = 10
epochs = 500
MCDO_amount_of_predictions = 500
MCDO_batch_size = 250
train_test_split = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
to_shuffle = True
augmentation = False
plot_imgs = True
label_normalizer = True
save_augmentation_to_hdf5 = True
add_dropout = True
MCDO = True
learn_rate = 0.0001


# input image dimensions
img_rows, img_cols, img_depth = 32, 32, 3

# loading the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(np.shape(x_train), np.shape(y_train))


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(np.shape(x_train), np.shape(y_train))

test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above



def get_dropout(input_tensor, p = 0.5, MCDO = False):
    if MCDO:
        return Dropout(p)(input_tensor, training = True)
    else:
        return Dropout(p)(input_tensor)


# mcdo_model.add(VGG16(include_top = False, weights = 'imagenet', input_shape = (img_rows, img_cols, img_depth), add_dropout = True, MCDO = True))

from tensorflow.keras.applications.vgg16 import VGG16
mcdo_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_depth))

def insert_intermediate_layer_in_keras(model, layer_id, p):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = get_dropout(x, p, MCDO = True)
        x = layers[i](x)
    
    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model

if add_dropout == True:
    p_list = [0.2, 0.25, 0.3, 0.35, 0.4]
    P_i = 0  
    layer_id = 1
    # Creating dictionary that maps layer names to the layers
    # layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])
    layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])

    for layer_name in layer_dict:
        layer_dict = dict([(layer.name, layer) for layer in mcdo_model.layers])
        if layer_name.endswith('_pool'):
            print(layer_name)
            layer_index = list(layer_dict).index(layer_name)
            print(layer_index)
            # Add a dropout (trainable) layer
            mcdo_model = insert_intermediate_layer_in_keras(mcdo_model, layer_index + 1, p_list[P_i])
            P_i += 1


            # mcdo_model.summary()

    # Stacking a new simple convolutional network on top of it   
    layers = [l for l in mcdo_model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        x = layers[i](x)

    x = get_dropout(x, p_list[P_i - 1], MCDO = True)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)


    # Creating new model. Please note that this is NOT a Sequential() model.
    mcdo_model = Model(inputs=layers[0].input, outputs=x)



for layer in mcdo_model.layers:
    layer.trainable = True

mcdo_model.summary()

adam = optimizers.Adam(lr = learn_rate)
mcdo_model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Start fitting monte carlo dropout model")

os.chdir(fig_dir)
logs_dir="/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

h_mc = mcdo_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test), 
                    callbacks=[tensorboard_callback])

mc_predictions = []

progress_bar = tf.keras.utils.Progbar(target=MCDO_amount_of_predictions,interval=5)
for i in range(MCDO_amount_of_predictions):
    progress_bar.update(i)
    y_p = mcdo_model.predict(x_test, batch_size=MCDO_batch_size)
    mc_predictions.append(y_p)

# score of the mc model
accs = []
for y_p in mc_predictions:
    acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
    accs.append(acc)
print("MC accuracy: {:.1%}".format(sum(accs)/len(accs)))

mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
ensemble_acc = accuracy_score(y_test.argmax(axis=1), mc_ensemble_pred)
print("MC-ensemble accuracy: {:.1%}".format(ensemble_acc))

tf.confusion_matrix(y_test.argmax(axis=1), mc_ensemble_pred)

plt.hist(accs)
plt.axvline(x=ensemble_acc, color="b")
plt.savefig('ensemble_acc.png')
plt.clf()

plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])


p0 = np.array([p[test_img_idx] for p in mc_predictions])
print("posterior mean: {}".format(p0.mean(axis=0).argmax()))
print("true label: {}".format(y_test[test_img_idx].argmax()))
print()
# probability + variance
for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
    print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))


x, y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
plt.plot(x, y)
plt.savefig('prob_var_' + str(test_img_idx) + '.png')
plt.clf()

fig, axes = plt.subplots(5, 1, figsize=(12,6))

for i, ax in enumerate(fig.get_axes()):
    ax.hist(p0[:,i], bins=100, range=(0,1))
    ax.set_title(f"class {i}")
    ax.label_outer()

fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)