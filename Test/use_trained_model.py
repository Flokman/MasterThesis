from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import os, datetime, time
import h5py
import random
from random import seed, randint, shuffle
import glob
import re
import cv2


from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

batch_size = 32
num_classes = 5
epochs = 100
amount_of_predictions = 500
batch_size = 250
train_test_split = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
to_shuffle = True
augmentation = False
plot_imgs = True
label_normalizer = True
save_augmentation_to_hdf5 = True
add_batch_normalization = True
add_batch_normalization_inside = True
train_all_layers = True
only_after_specific_layer = True
weights_to_use = None
learn_rate = 0.001

load_trained_model = False
model_to_use = '/Dropout'
hdf5_dataset = False
dataset_loc = '/test_images'
dataset_name = ''
labels_avail = False
img_rows, img_cols, img_depth = 256,  256, 3 # target image size to resize to

dir_path_head_tail = os.path.split(os.path.dirname(os.path.realpath(__file__)))
root_path = dir_path_head_tail[0] 

if model_to_use == '/Dropout':
    model_version = '/None_Yes_Retrain_64_73'
    model_name = 'mcdo_model.h5'
elif model_to_use == '/BN':
    model_version = ''
    model_name = 'MCBN_model.h5'

# Get model path
data_path = root_path + model_to_use + model_version + model_name

# Get dataset path
if hdf5_dataset == True:
    # Input image dimensions
    img_rows, img_cols, img_depth = 256,  256, 3
    dataset_loc = '/Datasets'
    dataset_name = '/Messidor2_PNG_AUG_' + str(img_rows) + '.hdf5'
    data_path = root_path + dataset_loc + dataset_name
    
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test), label_count  = load_data(data_path, train_test_split, to_shuffle)
    
    test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

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

else:
    data_path = os.getcwd() + dataset_loc
    print(data_path)

    # get all the image paths 
    addrs = glob.glob(images_path)
    x.shape = (len(addrs), img_rows, img_cols, img_depth)
    x[:,0] = np.arange(len(addrs))

    for i in range(len(addrs)):
        if i % 1000 == 0 and i > 1:
            print ('Image data: {}/{}'.format(i, len(addrs)) )

        addr = addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)# resize to (img_rows, img_cols)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
        x[i, ...] = img[None]
        print(x[i])

    if labels_avail == True: 
        baseaddrs = []
        for im in addrs:
            if im.endswith('JPG'):
                im = im[:-3]
                im = im + 'jpg'
            baseaddrs.append(str(os.path.basename(im)))

        labels = []
        label_count = [0] * 5
        with open(dir_path + dataset_loc + dataset_loc + '.csv') as f:
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
        y.shape = (len(labels), len(labels))
        y[:,0] = np.arange(len(labels))
        y[0,:] = labels
        combined = list(zip(x, y)) # use zip() to bind the images and labels together    
        (x, y) = zip(*combined)
        # convert class vectors to binary class matrices
        y = tf.keras.utils.to_categorical(y_train, num_classes)

    
    test_img_idx =  randint(0, len(x)) # For evaluation, this image is put in the fig_dir created above

    x = x.astype('float32')
    x /= 255
    print('x shape:', x.shape)
    print(x.shape[0], 'x samples')

def shuffle_data(x_to_shuff, y_to_shuff):
    combined = list(zip(x_to_shuff, y_to_shuff)) # use zip() to bind the images and label together
    random_seed = random.seed()
    print("Random seed for replication: {}".format(random_seed))
    random.shuffle(combined, random_seed)
 
    (x, y) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "x" then contains all the shuffled images and 
                               # "y" contains all the shuffled labels.
    return (x, y)


def load_data(path, train_test_split, to_shuffle):
    with h5py.File(data_path, "r") as f:
        (x, y) = np.array(f['x']), np.array(f['y'])
    label_count = [0] * num_classes
    for lab in y:
        label_count[lab] += 1

    if to_shuffle == True:
        (x, y) = shuffle_data(x, y)


    # Divide the data into a train and test set
    x_train = x[0:int(train_test_split*len(x))]
    y_train = y[0:int(train_test_split*len(y))]

    x_test = x[int(train_test_split*len(x)):]
    y_test = y[int(train_test_split*len(y)):]

    return (x_train, y_train), (x_test, y_test), label_count



old_dir = os.getcwd()
os.chdir(model_loc)
pre_trained_model = load_model(model_name)
pre_trained_model.summary()

os.chdir(old_dir)
fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # Dir to store created figures
os.makedirs(fig_dir)
log_dir = os.path.join(fig_dir,"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Dir to store Tensorboard data
os.makedirs(log_dir)
os.chdir(fig_dir)


mc_predictions = []

progress_bar = tf.keras.utils.Progbar(target=amount_of_predictions,interval=5)
for i in range(amount_of_predictions):
    progress_bar.update(i)
    y_p = pre_trained_model.predict(x_test, batch_size=batch_size)
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

confusion = tf.confusion_matrix(labels = y_test.argmax(axis=1), predictions = mc_ensemble_pred, num_classes = num_classes)
sess = tf.Session()
with sess.as_default():
        print(sess.run(confusion))

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