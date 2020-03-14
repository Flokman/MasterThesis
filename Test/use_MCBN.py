''' Returns the prediction and uncertainty of images in /test_images of a pretrained
(batch normarmalization) network of choice '''

import os
import random
import glob
import re
import csv
import h5py
import tensorflow as tf
import numpy as np
# import astroNN

# from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from astroNNcustom import MCBatchNorm

from tensorflow.keras import Input, layers, models, utils, backend

# Hyperparameters
NUM_CLASSES = 5
MCBN_PREDICTIONS = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TO_SHUFFLE = True
LEARN_RATE = 0.001
MODEL_TO_USE = os.path.sep + 'MCBN'
MODEL_VERSION = os.path.sep + '2020-02-26_11-48-44'
MODEL_NAME = 'MCBN_model.h5'
HDF5_DATASET = True
MINIBATCH_SIZE = 128
TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'
DATASET_NAME = ''
LABELS_AVAILABLE = False
WEIGHTS_TO_USE = None
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to


WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0] 

ONLY_AFTER_SPECIFIC_LAYER = True
ADD_BATCH_NORMALIZATION_INSIDE = True
MCBN = True


def shuffle_data(x_to_shuff, y_to_shuff):
    ''' Shuffle the data randomly '''
    combined = list(zip(x_to_shuff, y_to_shuff)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
 
    (x, y) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "x" then contains all the shuffled images and
                               # "y" contains all the shuffled labels.
    return (x, y)


def load_data(path, train_test_split, to_shuffle):
    ''' Load a dataset from a hdf5 file '''
    with h5py.File(path, "r") as f:
        (x, y) = np.array(f['x']), np.array(f['y'])
    if to_shuffle:
        (x, y) = shuffle_data(x, y)

    # Divide the data into a train and test set
    x_train = x[0:int(train_test_split*len(x))]
    y_train = y[0:int(train_test_split*len(y))]

    x_test = x[int(train_test_split*len(x)):]
    y_test = y[int(train_test_split*len(y)):]

    return (x_train, y_train), (x_test, y_test)


def load_hdf5_dataset():
    ''' Load a dataset, split and put in right format'''
    dataset_location = os.path.sep + 'Datasets'
    dataset_name = os.path.sep + 'Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
    data_path = ROOT_PATH + dataset_location + dataset_name
    
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test)  = load_data(data_path, TRAIN_TEST_SPLIT, TO_SHUFFLE)
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return((x_train, y_train), (x_test, y_test))


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
            x = img_ar.reshape((1,) + img_ar.shape)
        
        else:
            img = load_img(addr, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_ar = img_to_array(img)
            img_ar = img_ar.reshape((1,) + img_ar.shape)
            x = np.vstack((x, img_ar))

    x = x.astype('float32')
    x /= 255
    print('x shape:', x.shape)
    print(x.shape[0], 'x samples')

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
        y.shape = (len(labels), len(labels))
        y[:,0] = np.arange(len(labels))
        y[0,:] = labels
        combined = list(zip(x, y)) # use zip() to bind the images and labels together    
        (x, y) = zip(*combined)
        # convert class vectors to binary class matrices
        y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
        
        return(x,y)
    else:
        return(x)


def create_minibatch(x, y):
    ''' Returns a minibatch of the train data '''
    combined = list(zip(x, y)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    # print("Random seed minibatch for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
    minibatch = combined[:MINIBATCH_SIZE]
    
    (x_minibatch, y_minibatch) = zip(*minibatch)  # *combined is used to separate all the tuples in the list combined,
                            # "x_minibatch" then contains all the shuffled images and
                            # "y_minibatch" contains all the shuffled labels.

    return(x_minibatch, y_minibatch)


def MCBNVGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=NUM_CLASSES,
          **kwargs):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MCBatchNorm(x)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MCBatchNorm(x)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MCBatchNorm(x)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MCBatchNorm(x)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = MCBatchNorm(x)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MCBatchNorm(x)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = MCBatchNorm(x)(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = MCBatchNorm(x)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def main():
    ''' Main function '''
    # Get dataset path
    if HDF5_DATASET:
        (x_train, y_train), (x_test, y_test) = load_hdf5_dataset()

    if LABELS_AVAILABLE:
        (x_pred, y_pred) = load_new_images()
    else:
        x_pred = load_new_images()

    old_dir = os.getcwd()
    os.chdir(ROOT_PATH + MODEL_TO_USE + MODEL_VERSION + os.path.sep)
    print(os.getcwd())

    # Reload the model from the 2 files we saved
    with open('MCBN_model_config.json') as json_file:
        json_config = json_file.read()
    pre_trained_model = tf.keras.models.model_from_json(json_config)
    pre_trained_model.load_weights('MCBN_weights_{}.h5'.format(i))
    # pre_trained_model = MCBNVGG16(weights=None, include_top=True,
    #                        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    
    # pre_trained_model.load_weights('path_to_my_weights.h5')


    # Set onoly batch normalization layers to trainable
    for layer in pre_trained_model.layers:
        if re.search('BatchNormalization.*', layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
        print(layer.name, layer.trainable)

    adam = optimizers.Adam(lr=LEARN_RATE)
    pre_trained_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    pre_trained_model.summary()

    ##### https://github.com/fizyr/keras-retinanet/issues/214 ####
    
    mc_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)

    os.chdir(old_dir)
    org_model = pre_trained_model
    for i in range(MCBN_PREDICTIONS):
        progress_bar.update(i)
        # Create new random minibatch from train data
        x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
        x_minibatch = np.asarray(x_minibatch)
        y_minibatch = np.asarray(y_minibatch)

        # Fit the BN layers with the new minibatch, leave all other weights the same
        pre_trained_model = org_model
        pre_trained_model.fit(x_minibatch, y_minibatch,
                            batch_size=MINIBATCH_SIZE,
                            epochs=1,
                            verbose=0)

        y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred)) #Predict for bn look at (sigma and mu only one to chance, not the others)
        mc_predictions.append(y_p)
        # print(y_p)


    # for i in range(MCBN_PREDICTIONS):
    #     progress_bar.update(i)
    #     y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred))
    #     mc_predictions.append(y_p)

    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in mc_predictions])
        print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
            print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))

if __name__== "__main__":
    main()
