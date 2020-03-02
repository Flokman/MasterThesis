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
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np
import pandas as pd
import os, datetime, time
import h5py
import random
from random import seed, randint, shuffle
import glob
import re

from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

batch_size = 32
num_classes = 5
epochs = 100
amount_of_predictions = 50
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
model_to_use = os.path.sep + 'BN'
hdf5_dataset = True
MCBN_data = True
minibatch_size = 64
dataset_loc = os.path.sep + 'test_images'
dataset_name = ''
labels_avail = False
img_height, img_width, img_depth = 256,  256, 3 # target image size to resize to

dir_path_head_tail = os.path.split(os.path.dirname(os.path.realpath(__file__)))
root_path = dir_path_head_tail[0] 

if model_to_use == '/Dropout':
    model_version = '/None_Yes_Retrain_64_73'
    model_name = 'mcdo_model.h5'
elif model_to_use == (os.path.sep + 'BN'):
    model_version = os.path.sep + '2020-01-24_16-21-21'
    model_name = 'MCBN_model.h5'

# Get model path
model_path = root_path + model_to_use + model_version + model_name
# print(model_path)

def shuffle_data(x_to_shuff, y_to_shuff):
    combined = list(zip(x_to_shuff, y_to_shuff)) # use zip() to bind the images and label together
    random_seed = random.randint(0,1000)
    print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
 
    (x, y) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "x" then contains all the shuffled images and 
                               # "y" contains all the shuffled labels.
    return (x, y)


def load_data(path, train_test_split, to_shuffle):
    with h5py.File(path, "r") as f:
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


def load_hdf5_dataset():
    # Input image dimensions
    img_height, img_width, img_depth = 256,  256, 3
    dataset_loc = os.path.sep + 'Datasets'
    dataset_name = os.path.sep + 'Messidor2_PNG_AUG_' + str(img_height) + '.hdf5'
    data_path = root_path + dataset_loc + dataset_name
    
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test), label_count  = load_data(data_path, train_test_split, to_shuffle)
    
    test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_depth, img_height, img_width)
        x_test = x_test.reshape(x_test.shape[0], img_depth, img_height, img_width)
        input_shape = (img_depth, img_height, img_width)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_height, img_width, img_depth)
        x_test = x_test.reshape(x_test.shape[0], img_height, img_width, img_depth)
        input_shape = (img_height, img_width, img_depth)

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

    return((x_train, y_train), (x_test, y_test))


def load_new_images():
    images_path = os.getcwd() + dataset_loc + os.path.sep + '*'
    print(images_path)

    # get all the image paths 
    addrs = glob.glob(images_path)

    for i in range(len(addrs)):
        if i % 1000 == 0 and i > 1:
            print ('Image data: {}/{}'.format(i, len(addrs)) )

        if i == 0:
            addr = addrs[i]
            img = load_img(addr, target_size = (img_height, img_width))
            img_ar = img_to_array(img)
            x = img_ar.reshape((1,) + img_ar.shape)
        
        else:
            addr = addrs[i]
            img = load_img(addr, target_size = (img_height, img_width))
            img_ar = img_to_array(img)
            img_ar = img_ar.reshape((1,) + img_ar.shape)
            x = np.vstack((x,img_ar))

    
    test_img_idx =  randint(0, len(x)) # For evaluation, this image is put in the fig_dir created above
    x = x.astype('float32')
    x /= 255
    print('x shape:', x.shape)
    print(x.shape[0], 'x samples')

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
        
        return(x,y)
    else:
        return(x)


def create_minibatch(x, y):
    combined = list(zip(x, y)) # use zip() to bind the images and label together
    random_seed = random.randint(0,1000)
    print("Random seed minibatch for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
    minibatch = combined[:minibatch_size]
    

    (x_minibatch, y_minibatch) = zip(*minibatch)  # *combined is used to separate all the tuples in the list combined,  
                            # "x_minibatch" then contains all the shuffled images and 
                            # "y_minibatch" contains all the shuffled labels. 
    # x_minibatch = np.array(x_minibatch)
    # print(x_minibatch.shape)
    return(x_minibatch, y_minibatch)

# Get dataset path
if hdf5_dataset == True:
    (x_train, y_train), (x_test, y_test) = load_hdf5_dataset()

    if MCBN_data == True:
        if labels_avail == True:
            (x_pred, y_pred) = load_new_images()
        else:
            x_pred = load_new_images()

else:
    if labels_avail == True:
        (x_pred, y_pred) = load_new_images()
    else:
        x_pred = load_new_images()      



old_dir = os.getcwd()
os.chdir(root_path + model_to_use + model_version + os.path.sep)
print(os.getcwd())
pre_trained_model = load_model(model_name)

# Set model layers to untrainable
# for layer in pre_trained_model.layers:
#     layer.trainable = False

layer_dict = dict([(layer.name, layer) for layer in pre_trained_model.layers])
for layer in pre_trained_model.layers:
    if re.search('.*_normalization.*', layer.name):
        layer.trainable = True
    else:
        layer.trainable = False

for l in pre_trained_model.layers:
    print(l.name, l.trainable)

adam = optimizers.Adam(lr = learn_rate)
pre_trained_model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

pre_trained_model.summary()

os.chdir(old_dir)
fig_dir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # Dir to store created figures
os.makedirs(fig_dir)
log_dir = os.path.join(fig_dir,"logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Dir to store Tensorboard data
os.makedirs(log_dir)
os.chdir(fig_dir)
logs_dir= os.path.sep +"logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

if MCBN_data == True:
    mc_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=amount_of_predictions,interval=5)
    for i in range(amount_of_predictions):
        progress_bar.update(i)
        # Create new random minibatch from train data
        x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
        x_minibatch = np.asarray(x_minibatch)
        y_minibatch = np.asarray(y_minibatch)

        # Fit the BN layers with the new minibatch, leave all other weights the same

        pre_trained_model.fit(x_minibatch, y_minibatch,
                            batch_size=minibatch_size,
                            epochs=1,
                            verbose=2, 
                            callbacks=[tensorboard_callback])

        y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred))
        mc_predictions.append(y_p)
        

for i in range(len(x_pred)):
    p0 = np.array([p[i] for p in mc_predictions])
    print("posterior mean: {}".format(p0.mean(axis=0).argmax()))
    # probability + variance
    for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
        print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))

# # score of the mc model
# accs = []
# for y_p in mc_predictions:
#     acc = accuracy_score(y_pred.argmax(axis=1), y_p.argmax(axis=1))
#     accs.append(acc)
# print("MC accuracy: {:.1%}".format(sum(accs)/len(accs)))

# mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
# ensemble_acc = accuracy_score(y_pred.argmax(axis=1), mc_ensemble_pred)
# print("MC-ensemble accuracy: {:.1%}".format(ensemble_acc))

# confusion = tf.confusion_matrix(labels = y_pred.argmax(axis=1), predictions = mc_ensemble_pred, num_classes = num_classes)
# sess = tf.Session()
# with sess.as_default():
#         print(sess.run(confusion))

# plt.hist(accs)
# plt.axvline(x=ensemble_acc, color="b")
# plt.savefig('ensemble_acc.png')
# plt.clf()

# plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])


# p0 = np.array([p[test_img_idx] for p in mc_predictions])
# print("posterior mean: {}".format(p0.mean(axis=0).argmax()))
# print("true label: {}".format(y_test[test_img_idx].argmax()))
# print()
# # probability + variance
# for i, (prob, var) in enumerate(zip(p0.mean(axis=0), p0.std(axis=0))):
#     print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))


# x, y = list(range(len(p0.mean(axis=0)))), p0.mean(axis=0)
# plt.plot(x, y)
# plt.savefig('prob_var_' + str(test_img_idx) + '.png')
# plt.clf()

# fig, axes = plt.subplots(5, 1, figsize=(12,6))

# for i, ax in enumerate(fig.get_axes()):
#     ax.hist(p0[:,i], bins=100, range=(0,1))
#     ax.set_title(f"class {i}")
#     ax.label_outer()

# fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)