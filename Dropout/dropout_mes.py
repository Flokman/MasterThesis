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

def plot_images(dataset, n_images, samples_per_image, plotname):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.savefig(plotname + '.png')
    plt.clf()

# Experimental
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

# Experimantal stop

def data_augmentation(x_data, y_data, label_count, label_normalizer):
    #x_data = (x_data / 255).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors((x_data, y_data))
    
    # Add augmentations
    augmentations = [flip, color, zoom, rotate]
    for f in augmentations:
        # Apply augmentation in % of cases
        dataset = dataset.map(lambda self, x: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=multiprocessing_cpu_count)
    dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1))
    if plot_imgs == True:
        plot_dataset = tf.data.Dataset.from_tensor_slices((x[0:8] / 255).astype(np.float32))
        plot_images(plot_dataset, 8, 10, 'post_aug')
    return tf.data.Dataset.as_numpy(dataset)

def load_data(path, train_test_split, data_augmentation, to_shuffle):
    with h5py.File(data_path, "r") as f:
        x, y = np.array(f['x']), np.array(f['y'])

    label_count = [0] * num_classes
    for lab in y:
        label_count[lab] += 1

    print(x.shape, y.shape)
    x = x.astype('float32')
    x /= 255
    y = tf.keras.utils.to_categorical(y, num_classes)

    if to_shuffle == True:
        x, y = shuffle_data(x, y)

    if augmentation == True:
        (x, y) = data_augmentation(x, y, label_count, label_normalizer)

    # Divide the data into a train and test set
    x_train = x[0:int(train_test_split*len(x))]
    y_train = y[0:int(train_test_split*len(y))]

    x_test = x[int(train_test_split*len(x)):]
    y_test = y[int(train_test_split*len(y)):]

    return (x_train, y_train), (x_test, y_test), label_count

# Split the data between train and test sets
(x_train, y_train), (x_test, y_test), label_count  = load_data(data_path, train_test_split, data_augmentation, to_shuffle)

test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above

print("dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {}, MCDO_amount_of_predictions = {}, MCDO_batch_size = {}, test_img_idx = {}, train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {}, label_normalizer = {}".format(dataset_name, batch_size, num_classes, epochs, MCDO_amount_of_predictions, MCDO_batch_size, test_img_idx, train_test_split, to_shuffle, augmentation, label_count, label_normalizer))


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