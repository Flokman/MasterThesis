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
import astroNN

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from astroNN import MCBatchNorm

# Hyperparameters
NUM_CLASSES = 5
MCBN_PREDICTIONS = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TO_SHUFFLE = True
LEARN_RATE = 0.001
MODEL_TO_USE = os.path.sep + 'BN'
MODEL_VERSION = os.path.sep + '2020-02-18_12-06-47'
MODEL_NAME = 'MCBN_model.h5'
HDF5_DATASET = True
MINIBATCH_SIZE = 128
TEST_IMAGES_LOCATION = os.path.sep + 'test_images'
TEST_IMAGES_LABELS_NAME = 'test_images_labels'
DATASET_NAME = ''
LABELS_AVAILABLE = False
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3 # target image size to resize to

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


def get_mcbn(input_tensor):
    ''' Returns a mcbn layer with probability prob, either trainable or not '''
    if MCBN:
        return astroNN.MCBatchNorm()(input_tensor)
    else:
        return BatchNormalization()(input_tensor)


def insert_intermediate_layer_in_keras(model, layer_id):
    ''' Insert a mcbn layer before the layer with layer_id '''
    inter_layers = [l for l in model.layers]

    x = inter_layers[0].output
    for i in range(1, len(inter_layers)):
        if i == layer_id:
            x = get_mcbn(x)
        x = inter_layers[i](x)

    new_model = Model(inputs=inter_layers[0].input, outputs=x)
    return new_model


def add_batch_normalization(mcbn_model):
    ''' Adds batch normalizaiton layers either after all pool and dense layers
        or only after dense layers '''
    if ADD_BATCH_NORMALIZATION_INSIDE:
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in mcbn_model.layers])

        for layer_name in layer_dict:
            layer_dict = dict([(layer.name, layer) for layer in mcbn_model.layers])
            if ONLY_AFTER_SPECIFIC_LAYER:
                if re.search('.*_conv.*', layer_name):
                    print(layer_name)
                    layer_index = list(layer_dict).index(layer_name)
                    print(layer_index)

                    # Add a batch normalization (trainable) layer
                    mcbn_model = insert_intermediate_layer_in_keras(mcbn_model, layer_index + 1)

                    # mcbn_model.summary()
            else:
                print(layer_name)
                layer_index = list(layer_dict).index(layer_name)
                print(layer_index)

                # Add a batch normalization (trainable) layer
                mcbn_model = insert_intermediate_layer_in_keras(mcbn_model, layer_index + 1)



        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcbn_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        # x = get_mcbn(x)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_mcbn(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_mcbn(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    else:
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in mcbn_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_mcbn(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_mcbn(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Creating new model
    mcbn_model = Model(inputs=all_layers[0].input, outputs=x)
    return mcbn_model


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
    # pre_trained_model = tf.keras.models.load_model(MODEL_NAME, custom_objects={'MCBatchNorm': MCBatchNorm})
    
    # # Reload the model from the 2 files we saved
    # with open('mcbn_model_config.json') as json_file:
    #     json_config = json_file.read()
    # pre_trained_model = keras.models.model_from_json(json_config, custom_objects={'MCBatchNorm': MCBatchNorm})
    # pre_trained_model.load_weights('path_to_my_weights.h5')

    # VGG16 since it does not include batch normalization of mcbn by itself
    pre_trained_model = VGG16(weights=None, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    pre_trained_model = add_batch_normalization(pre_trained_model)
    


    # Set batch normalization layers to untrainable
    for layer in pre_trained_model.layers:
        if re.search('.*_normalization.*', layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
        # print(layer.name, layer.trainable)

    adam = optimizers.Adam(lr=LEARN_RATE)
    pre_trained_model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    pre_trained_model.summary()
    pre_trained_model.load_weights('path_to_my_weights.h5', by_name=True)
    ##### https://github.com/fizyr/keras-retinanet/issues/214 ####
    
    mc_predictions = []
    progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)

    # with tf.Session() as sess:
    #     # Initialise all variables
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     pre_trained_model.load_weights('path_to_my_weights.h5')
    #     os.chdir(old_dir)
    #     for i in range(MCBN_PREDICTIONS):
    #         progress_bar.update(i)
    #         # Create new random minibatch from train data
    #         x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
    #         x_minibatch = np.asarray(x_minibatch)
    #         y_minibatch = np.asarray(y_minibatch)

    #         # Fit the BN layers with the new minibatch, leave all other weights the same

    #         pre_trained_model.fit(x_minibatch, y_minibatch,
    #                             batch_size=MINIBATCH_SIZE,
    #                             epochs=1,
    #                             verbose=0)

    #         y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred)) #Predict for bn look at (sigma and mu only one to chance, not the others)
    #         mc_predictions.append(y_p)


    for i in range(MCBN_PREDICTIONS):
        progress_bar.update(i)
        y_p = pre_trained_model.predict(x_pred, batch_size=len(x_pred))
        mc_predictions.append(y_p)

    for i in range(len(x_pred)):
        p_0 = np.array([p[i] for p in mc_predictions])
        print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
        # probability + variance
        for l, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
            print("class: {}; proba: {:.1%}; var: {:.2%} ".format(l, prob, var))

if __name__== "__main__":
    main()
