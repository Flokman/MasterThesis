''' Trains a (pre trained) network with additional batch normalization layers
    for uncertainty estimation'''

import os
import datetime
import time
import random
import re
import h5py
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 32, 32, 3
DATASET_NAME = 'CIFAR10'

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 50
MCBN_PREDICTIONS = 25
MINIBATCH_SIZE = BATCH_SIZE
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
SAVE_AUGMENTATION_TO_HDF5 = False
ADD_BATCH_NORMALIZATION = True
ADD_BATCH_NORMALIZATION_INSIDE = True

TO_SHUFFLE = False
AUGMENTATION = False
LABEL_NORMALIZER = False
TRAIN_ALL_LAYERS = True

ONLY_AFTER_SPECIFIC_LAYER = True
WEIGHTS_TO_USE = 'imagenet'
LEARN_RATE = 0.001
ES_PATIENCE = 15
MIN_DELTA = 0.005
EARLY_MONITOR = 'val_loss'
MC_MONITOR = 'val_loss'

adam = optimizers.Adam(lr=LEARN_RATE)
sgd = optimizers.SGD(lr=LEARN_RATE, momentum=0.9)

OPTIMZ = sgd

RESULTFOLDER = 'CIFAR10'

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ROOT_PATH = DIR_PATH_HEAD_TAIL[0]
DATA_PATH = ROOT_PATH + '/Datasets' + DATASET_NAME


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {},
        MCBN_PREDICTIONS = {}, Mini_batch_size = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {},
        label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        add_bn_inside = {}, train_all_layers = {}, weights_to_use = {},
        es_patience = {}, train_val_split = {}, MIN_DELTA = {}, Early_monitor = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS,
            MCBN_PREDICTIONS, MINIBATCH_SIZE, test_img_idx,
            TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION,
            LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
            ADD_BATCH_NORMALIZATION_INSIDE, TRAIN_ALL_LAYERS, WEIGHTS_TO_USE,
            ES_PATIENCE, TRAIN_VAL_SPLIT, MIN_DELTA, EARLY_MONITOR))

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return(x_train, y_train, x_test, y_test, test_img_idx)


def get_batch_normalization(input_tensor):
    ''' Returns a trainable batch normalization layer '''
    return BatchNormalization()(input_tensor)


def insert_intermediate_layer_in_keras(model, layer_id):
    ''' Insert a batch normalization layer before the layer with layer_id '''
    inter_layers = [l for l in model.layers]

    x = inter_layers[0].output
    for i in range(1, len(inter_layers)):
        if i == layer_id:
            x = get_batch_normalization(x)
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
        # x = get_batch_normalization(x)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = get_batch_normalization(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_batch_normalization(x)
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
        x = get_batch_normalization(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = get_batch_normalization(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Creating new model
    mcbn_model = Model(inputs=all_layers[0].input, outputs=x)
    return mcbn_model


def create_minibatch(x, y):
    ''' Returns a minibatch of the train data '''
    combined = list(zip(x, y)) # use zip() to bind the images and label together
    random_seed = random.randint(0, 1000)
    # print("Random seed minibatch for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)
    minibatch = combined[:MINIBATCH_SIZE]

    (x_minibatch, y_minibatch) = zip(*minibatch)  
                            # *combined is used to separate all the tuples in the list combined,
                            # "x_minibatch" then contains all the shuffled images and
                            # "y_minibatch" contains all the shuffled labels.

    return(x_minibatch, y_minibatch)


def reset_weights(model):
    for layer in model.layers:
        if re.search('batch_normalization_.*', layer.name):
            print(layer.name)  
            if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
                reset_weights(layer) #apply function recursively
                continue

            #where are the initializers?
            if hasattr(layer, 'cell'):
                init_container = layer.cell
            else:
                init_container = layer

            for key, initializer in init_container.__dict__.items():
                if "initializer" not in key: #is this item an initializer?
                    continue #if no, skip it
                    print("skip")

                # find the corresponding variable, like the kernel or the bias
                if key == 'recurrent_initializer': #special case check
                    var = getattr(init_container, 'recurrent_kernel')
                else:
                    var = getattr(init_container, key.replace("_initializer", ""))

                var.assign(initializer(var.shape, var.dtype))
                # print("reinialized")
                #use the initializer

    return model


def mcbn_predict(MCBN_model, x_train, y_train, x_test, y_test):
    mcbn_predictions = []
    
    # with open('MCBN_model_config.json') as json_file:
    #     json_config = json_file.read()
    # MCBN_test_model = tf.keras.models.model_from_json(json_config)
    org_model = MCBN_model

    progress_bar = tf.keras.utils.Progbar(target=MCBN_PREDICTIONS, interval=5)
    for i in range(0, MCBN_PREDICTIONS):
        progress_bar.update(i)
        datagen = ImageDataGenerator(rescale=1./255)

        # Create new random minibatch from train data
        x_minibatch, y_minibatch = create_minibatch(x_train, y_train)
    
        x_minibatch = np.asarray(x_minibatch)
        # y_minibatch = np.asarray(y_minibatch)
        # y_minibatch = tf.keras.utils.to_categorical(y_minibatch, NUM_CLASSES)

        
        minibatch_generator = datagen.flow(x_minibatch,
                                    y_minibatch,
                                    batch_size=MINIBATCH_SIZE)


        # # # # Reload the model from the 2 files we saved
        # # # MCBN_test_model.load_weights('MCBN_weights.h5')
       
        # # # MCBN_test_model = reset_weights(MCBN_test_model)

        # # # # Set only batch normalization layers to trainable
        # # # for layer in MCBN_test_model.layers:
        # # #     if re.search('batch_normalization.*', layer.name):
        # # #         layer.trainable = True
        # # #     else:
        # # #         layer.trainable = False
        # # #     # print(layer.name, layer.trainable)

        # # # MCBN_test_model.compile(
        # # #     optimizer=OPTIMZ,
        # # #     loss='categorical_crossentropy',
        # # #     metrics=['accuracy'] )


        MCBN_test_model = org_model
        # Fit the BN layers with the new minibatch, leave all other weights the same
        MCBN_test_model.fit(minibatch_generator,
                            epochs=1,
                            verbose=0)


        test_generator = datagen.flow(x_test, batch_size=BATCH_SIZE, shuffle=False)
        y_p = MCBN_test_model.predict(test_generator) #Predict for bn look at (sigma and mu only one to chance, not the others)
        mcbn_predictions.append(y_p)

        # y_predict = MCBN_test_model.predict(datagen.flow(x_test, batch_size=BATCH_SIZE)) #Predict for bn look at (sigma and mu only one to chance, not the others)
        # mcbn_predictions.append(y_predict)

        # accs = []
        # for j, image in enumerate(x_test):
        #     image = np.expand_dims(image, axis=0)
        #     pred = MCBN_test_model.predict(datagen.flow(image))
        #     hi_pred_ind = pred.argmax()
        #     ground_tru = y_test[j].argmax()
        #     if hi_pred_ind == ground_tru:
        #         accs.append(1)
        #     else:
        #         accs.append(0)

        # print("MCBN accuracy: {:.1%}".format(sum(accs)/len(accs)))

        # accs_new = []
        # for ind, pred in enumerate(mcbn_predictions):
        #     hi_pred_ind = pred.argmax()
        #     ground_tru = y_test[ind].argmax()
        #     if hi_pred_ind == ground_tru:
        #         accs_new.append(1)
        #     else:
        #         accs_new.append(0)

        # print("MCBN accuracy new: {:.1%}".format(sum(accs_new)/len(accs_new)))


        # y_predict = []
        # for j, image in enumerate(x_test):
        #     image = np.expand_dims(image, axis=0)
        #     pred = MCBN_test_model.predict(datagen.flow(image))
        #     y_predict.append(pred)
        
        # mcbn_predictions.append(y_predict)

        # y_predict = []
        # for j, image in enumerate(x_test):
        #     image = np.expand_dims(image, axis=0)
        #     pred = MCBN_test_model.predict(datagen.flow(image))
        #     y_predict.append(pred)
        
        # mcbn_predictions.append(y_predict)

        # score of the MCBN model
        # accs = []
        # for y_p in mcbn_predictions:
        #     acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
        #     accs.append(acc)
        # print("MCBN accuracy: {:.1%}".format(sum(accs)/len(accs)))

    return mcbn_predictions


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    # # VGG16 since it does not include batch normalization of dropout by itself
    # MCBN_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
    #                    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    # if ADD_BATCH_NORMALIZATION:
    #     MCBN_model = add_batch_normalization(MCBN_model)

    # else:
    #     # Stacking a new simple convolutional network on top of vgg16
    #     all_layers = [l for l in MCBN_model.layers]
    #     x = all_layers[0].output
    #     for i in range(1, len(all_layers)):
    #         x = all_layers[i](x)

    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation='relu', name='fc1')(x)
    #     x = Dense(4096, activation='relu', name='fc2')(x)
    #     x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    #     # Creating new model.
    #     MCBN_model = Model(inputs=all_layers[0].input, outputs=x)


    if WEIGHTS_TO_USE == None:
        # VGG16 since it does not include batch normalization of dropout by itself
        MCBN_model = VGG16(weights=WEIGHTS_TO_USE, include_top=True,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), classes = NUM_CLASSES)
    
    else:
        MCBN_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), layers=tf.keras.layers)
        # Stacking a new simple convolutional network on top of vgg16
        all_layers = [l for l in MCBN_model.layers]
        x = all_layers[0].output
        for i in range(1, len(all_layers)):
            x = all_layers[i](x)

        # Classification block
        # x = get_batch_normalization(x)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        # Creating new model
        MCBN_model = Model(inputs=all_layers[0].input, outputs=x)

        # MCBN_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
        #                 input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
        # # Stacking a new simple convolutional network on top of vgg16
        # all_layers = [l for l in MCBN_model.layers]
        # x = all_layers[0].output
        # for i in range(1, len(all_layers)):
        #     x = all_layers[i](x)

        # # Classification block
        # x = Flatten(name='flatten')(x)
        # x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        # # Creating new model
        MCBN_model = Model(inputs=all_layers[0].input, outputs=x)

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in MCBN_model.layers])

    for layer_name in layer_dict:
        layer_dict = dict([(layer.name, layer) for layer in MCBN_model.layers])
        if ONLY_AFTER_SPECIFIC_LAYER:
            if re.search('.*_conv.*', layer_name):
                layer_index = list(layer_dict).index(layer_name)

                # Add a batch normalization (trainable) layer
                MCBN_model = insert_intermediate_layer_in_keras(MCBN_model, layer_index + 1)

            if re.search('fc.*', layer_name):
                layer_index = list(layer_dict).index(layer_name)

                # Add a batch normalization (trainable) layer
                MCBN_model = insert_intermediate_layer_in_keras(MCBN_model, layer_index + 1)
            
            # if re.search('predictions', layer_name):
            #     print(layer_name)
            #     layer_index = list(layer_dict).index(layer_name)
            #     print(layer_index)

            #     # Add a batch normalization (trainable) layer
            #     MCBN_model = insert_intermediate_layer_in_keras(MCBN_model, layer_index + 1)
        else:
            layer_index = list(layer_dict).index(layer_name)

            # Add a batch normalization (trainable) layer
            MCBN_model = insert_intermediate_layer_in_keras(MCBN_model, layer_index + 1)


    if TRAIN_ALL_LAYERS:
        for layer in MCBN_model.layers:
            layer.trainable = True
            # print(layer, layer.trainable)

    elif ADD_BATCH_NORMALIZATION_INSIDE:
        for layer in MCBN_model.layers[:-6]:
            if re.search('batch_normalization.*', layer.name):
                layer.trainable = True
            else:
                layer.trainable = False
        #     layer.trainable = False
        # for layer in MCBN_model.layers:
        #     if re.search('batch_normalization.*', layer.name):
        #         layer.trainable = False
        #     # print(layer.name, layer.trainable)

    else:
        for layer in MCBN_model.layers:
            layer.trainable = False
        for layer in MCBN_model.layers:
            if re.search('batch_normalization.*', layer.name):
                layer.trainable = True
            if re.search('predictions', layer.name):
                layer.trainable = True

        # for layer in MCBN_model.layers:
        #     print(layer, layer.trainable)

    for layer in MCBN_model.layers:
        print(layer, layer.trainable)

    MCBN_model.summary()


    MCBN_model.compile(
        optimizer=OPTIMZ,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Start fitting monte carlo batch_normalization model")

    # Dir to store created figures
    fig_dir = os.path.join(os.getcwd(), RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=EARLY_MONITOR, min_delta = MIN_DELTA,
                                                    mode='auto', verbose=1, patience=ES_PATIENCE)

    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor=MC_MONITOR, mode='auto', save_best_only=True)

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size = BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)


    MCBN_model.fit(train_generator,
                   epochs=EPOCHS,
                   verbose=2,
                   validation_data=val_generator,
                   callbacks=[tensorboard_callback, early_stopping, mc])
    

    MCBN_model = load_model('best_model.h5')
    os.remove('best_model.h5')

    # Save JSON config to disk
    json_config = MCBN_model.to_json()
    with open('MCBN_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    MCBN_model.save_weights('MCBN_weights.h5')

    # test_loss, test_score = MCBN_model.evaluate(x_test, y_test)
    # print("Test Loss:", test_loss)
    # print("Test F1 Score:", test_score)

    # eval_gen = datagen.flow(x_test,
    #                             y_test,
    #                             batch_size = BATCH_SIZE)

    # first_eval = MCBN_model.evaluate(eval_gen)
    # print(first_eval)

    # Set only batch normalization layers to trainable
    for layer in MCBN_model.layers:
        if re.search('batch_normalization.*', layer.name):
            layer.trainable = True
        else:
            layer.trainable = False
        print(layer.name, layer.trainable)

    mcbn_predictions = mcbn_predict(MCBN_model, x_train, y_train, x_test, y_test)
    # print("MCBN accuracy: {:.1%}".format(sum(mcbn_evaluation)/len(mcbn_evaluation)))

    # mean = sum(mcbn_evaluation) / len(mcbn_evaluation)
    # variance = sum([((x - mean) ** 2) for x in mcbn_evaluation]) / len(mcbn_evaluation)
    # res = variance ** 0.5
    # print("variance and strdev :", variance, res)

    # score of the MCBN model
    accs = []
    for y_p in mcbn_predictions:
        acc = accuracy_score(y_test.argmax(axis=1), y_p.argmax(axis=1))
        accs.append(acc)
    print("MCBN accuracy: {:.1%}".format(sum(accs)/len(accs)))

    mcbn_ensemble_pred = np.array(mcbn_predictions).mean(axis=0).argmax(axis=1)
    ensemble_acc = accuracy_score(y_test.argmax(axis=1), mcbn_ensemble_pred)
    print("MCBN-ensemble accuracy: {:.1%}".format(ensemble_acc))

    dir_path_head_tail = os.path.split(os.path.dirname(os.getcwd()))
    if WEIGHTS_TO_USE != None:
        new_path = dir_path_head_tail[0] + os.path.sep + RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + WEIGHTS_TO_USE + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(ensemble_acc)
    else:
        new_path = dir_path_head_tail[0] + os.path.sep + RESULTFOLDER + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + 'None' + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(ensemble_acc)
    os.rename(fig_dir, new_path)

    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=mcbn_ensemble_pred,
                                    num_classes=NUM_CLASSES)
    print(confusion)


    plt.hist(accs)
    plt.axvline(x=ensemble_acc, color="b")
    plt.savefig('ensemble_acc.png')
    plt.clf()

    plt.imsave('test_image_' + str(test_img_idx) + '.png', x_test[test_img_idx])

    p_0 = np.array([p[test_img_idx] for p in mcbn_predictions])
    print("posterior mean: {}".format(p_0.mean(axis=0).argmax()))
    print("true label: {}".format(y_test[test_img_idx].argmax()))
    print()
    # probability + variance
    for i, (prob, var) in enumerate(zip(p_0.mean(axis=0), p_0.std(axis=0))):
        print("class: {}; proba: {:.1%}; var: {:.2%} ".format(i, prob, var))

    x_axis, y_axis = list(range(len(p_0.mean(axis=0)))), p_0.mean(axis=0)
    plt.plot(x_axis, y_axis)
    plt.savefig('prob_var_' + str(test_img_idx) + '.png')
    plt.clf()

    fig = plt.subplots(NUM_CLASSES, 1, figsize=(12, 6))[0]

    for i, ax in enumerate(fig.get_axes()):
        ax.hist(p_0[:, i], bins=100, range=(0, 1))
        ax.set_title(f"class {i}")
        ax.label_outer()

    fig.savefig('sub_plots' + str(test_img_idx) + '.png', dpi=fig.dpi)

if __name__ == "__main__":
    main()
