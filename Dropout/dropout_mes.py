from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import os, datetime
import h5py
from random import seed, randint, shuffle
import multiprocessing

multiprocessing_cpu_count = multiprocessing.cpu_count()

from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) # Dir to store created figures
os.makedirs(fig_dir)
log_dir = os.path.join(fig_dir,"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Dir to store Tensorboard data
os.makedirs(log_dir)

# Input image dimensions
img_rows, img_cols, img_depth = 256,  256, 3
dataset_name = '/Messidor2_PNG_' + str(img_rows) + '.hdf5'

batch_size = 32
num_classes = 5
epochs = 100
MCDO_amount_of_predictions = 500
MCDO_batch_size = 1000
train_test_split = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
to_shuffle = True
augmentation = True
plot_imgs = True
label_normalizer = True
save_augmentation_to_hdf5 = True

# Get dataset path
dir_path_head_tail = os.path.split(os.path.dirname(os.path.realpath(__file__)))
root_path = dir_path_head_tail[0] 
data_path = root_path + '/Datasets' + dataset_name

def shuffle_data(x_to_shuff, y_to_shuff):
    combined = list(zip(x_to_shuff, y_to_shuff)) # use zip() to bind the images and label together
    shuffle(combined)
 
    (x, y) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "x" then contains all the shuffled images and 
                               # "y" contains all the shuffled labels.
    return (x, y)


def data_augmentation(x, y, label_count, label_normalizer):
    # Importing necessary functions 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    # Initialising the ImageDataGenerator class. 
    # We will pass in the augmentation parameters in the constructor. 
    datagen = ImageDataGenerator( 
            rotation_range = 40, 
            shear_range = 0.2, 
            zoom_range = 0.2, 
            horizontal_flip = True, 
            brightness_range = (0.5, 1.5)) 
    
    if label_normalizer == True:
        from random import choices
        goal_amount = int(max(label_count) * 1.2) # Also perform some augmentation on the class with most examples 
        print("goal amount", goal_amount)
        
        # Sort by class
        sorted_tup = []
        for num in range(0, num_classes):
            sorted_tup.append([])
        for idx, label in enumerate(y):
            sorted_tup[label].append((x[idx], y[idx]))

        for label, label_amount in enumerate(label_count):
            amount_to_augment = int((goal_amount - label_amount)/5) # Divided by 5 since 5 augmentations will be performed at a time
            print("amount to aug", amount_to_augment*num_classes)
            print("class", label)
            tups_to_augment = choices(sorted_tup[label], k = amount_to_augment)
            
            to_augment_x = [(tups_to_augment[0])[0]]
            to_augment_y = np.empty(amount_to_augment)
            to_augment_y.fill(label)

            x_org_len = len(x)

            for i in range(1, amount_to_augment):
                to_augment_x = np.append(to_augment_x, [(tups_to_augment[i])[0]], axis=0)

            for i in range(0, amount_to_augment):
                i = 0
                # for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1, 
                #             save_to_dir ='preview/' + str(label),  
                #             save_prefix = str(label) + '_image', save_format ='png'):
                for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1):
                    x = np.append(x[:], batch[0], axis=0)
                    y = np.append(y[:], batch[1], axis=0)
                    print("{}/{}".format((len(x) - x_org_len), amount_to_augment*num_classes))
                    i += 1
                    if i > 5:
                        break

    if save_augmentation_to_hdf5 == True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hdf5_path = dir_path + '/Messidor2_PNG_AUG_' + str(img_rows) + '.hdf5'  # file path for the created .hdf5 file
        print(hdf5_path)

        # open a hdf5 file and create earrays 
        f = h5py.File(hdf5_path, mode='w')

        # PIL.Image: the pixels range is 0-255,dtype is uint.
        # matplotlib: the pixels range is 0-1,dtype is float.
        f.create_dataset("x", x.shape, np.uint8)

        # the ".create_dataset" object is like a dictionary, the "labels" is the key. 
        f.create_dataset("y", (len(x),), np.uint8)
        f["y"][...] = y

        # loop over train paths
        for i in range(len(x)):
        
            if i % 1000 == 0 and i > 1:
                print ('Image data: {}/{}'.format(i, len(x)) )

            f["x"][i, ...] = x[i]

            f.close()


    return (x, y)

def load_data(path, train_test_split, data_augmentation, to_shuffle):
    with h5py.File(data_path, "r") as f:
        x, y = np.array(f['x']), np.array(f['y'])

    label_count = [0] * num_classes
    for lab in y:
        label_count[lab] += 1

    if to_shuffle == True:
        x, y = shuffle_data(x, y)

    if augmentation == True:
        (x, y) = data_augmentation(x, y, label_count, label_normalizer)

    print("augmentation done")
    # Divide the data into a train and test set
    x_train = x[0:int(train_test_split*len(x))]
    y_train = y[0:int(train_test_split*len(y))]

    x_test = x[int(train_test_split*len(x)):]
    y_test = y[int(train_test_split*len(y)):]

    return (x_train, y_train), (x_test, y_test), label_count

# Split the data between train and test sets
(x_train, y_train), (x_test, y_test), label_count  = load_data(data_path, train_test_split, data_augmentation, to_shuffle)

test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above

print("dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {}, MCDO_amount_of_predictions = {}, MCDO_batch_size = {}, test_img_idx = {}, train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {}, label_normalizer = {}, save_augmentation_to_hdf5 = {}".format(dataset_name, batch_size, num_classes, epochs, MCDO_amount_of_predictions, MCDO_batch_size, test_img_idx, train_test_split, to_shuffle, augmentation, label_count, label_normalizer, save_augmentation_to_hdf5))


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_depth, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_depth, img_rows, img_cols)
    input_shape = (img_depth, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth)
    input_shape = (img_rows, img_cols, img_depth)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def create_model(mc=False, act="relu"):
    inp = Input(input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation=act)(inp)
    x = Conv2D(64, kernel_size=(2, 2), activation=act)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Flatten()(x)
    x = Dense(128, activation=act)(x)
    x = get_dropout(x, p=0.5, mc=mc)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

# model = create_model(mc=False, act="relu")

# h = model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(x_test, y_test))

# # score of the normal model
# score = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

print("Start fitting monte carlo dropout model")

mc_model = create_model(mc=True, act="relu")

os.chdir(fig_dir)
logs_dir="/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

h_mc = mc_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test), 
                    callbacks=[tensorboard_callback])



mc_predictions = []

progress_bar = tf.keras.utils.Progbar(target=MCDO_amount_of_predictions,interval=5)
for i in range(MCDO_amount_of_predictions):
    progress_bar.update(i)
    y_p = mc_model.predict(x_test, batch_size=MCDO_batch_size)
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