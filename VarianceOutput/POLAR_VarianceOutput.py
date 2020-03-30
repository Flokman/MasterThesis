''' Trains a (pre trained) network with additional dropout layers for uncertainty estimation'''

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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras import backend as K

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Input image dimensions
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH = 256, 256, 3
DATASET_NAME = '/Polar_PNG_' + str(IMG_HEIGHT) + '.hdf5'

BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS_1 = 120
ES_PATIENCE_1 = 3
EPOCHS_2 = 300
ES_PATIENCE_2 = 2

TEST_BATCH_SIZE = 250
TRAIN_TEST_SPLIT = 0.8 # Value between 0 and 1, e.g. 0.8 creates 80%/20% division train/test
TRAIN_VAL_SPLIT = 0.9
TO_SHUFFLE = False
AUGMENTATION = False
AUGMENT_TESTSET = False
LABEL_NORMALIZER = True
SAVE_AUGMENTATION_TO_HDF5 = False
TRAIN_ALL_LAYERS = True
WEIGHTS_TO_USE = None
LEARN_RATE = 0.00001
MIN_DELTA = 0.05
EARLY_MONITOR = 'loss'
DATANAME = 'POLAR'

# Get dataset path
DIR_PATH_HEAD_TAIL = os.path.split(os.path.dirname(os.path.realpath(__file__)))
ONE_HIGHER_PATH = os.path.split(DIR_PATH_HEAD_TAIL[0])
ROOT_PATH = ONE_HIGHER_PATH[0]
DATA_PATH = ROOT_PATH + '/Polar_dataset' + DATASET_NAME


def shuffle_data(x_to_shuff, y_to_shuff):
    ''' Shuffle the data randomly '''
    # use zip() to bind the images and label together
    combined = list(zip(x_to_shuff, y_to_shuff))
    random_seed = random.randint(0, 1000)
    print("Random seed for replication: {}".format(random_seed))
    random.seed(random_seed)
    random.shuffle(combined)

    (x_shuffled, y_shuffled) = zip(*combined)
        # *combined is used to separate all the tuples in the list combined,
        # "x_shuffled" then contains all the shuffled images and
        # "y_shuffled" contains all the shuffled labels.
    return (x_shuffled, y_shuffled)


def data_augmentation(x_aug, y_aug, label_count):
    ''' Augment the data according to the settings of imagedatagenerator '''
    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    if LABEL_NORMALIZER:
        start_aug = time.time()
        # Also perform some augmentation on the class with most examples
        goal_amount = int(max(label_count) * 1.2)
        print("goal amount", goal_amount)

        # Sort by class
        sorted_tup = []
        for num in range(0, NUM_CLASSES):
            sorted_tup.append([])
        for idx, label in enumerate(y_aug):
            sorted_tup[label].append((x_aug[idx], y_aug[idx]))

        for label, label_amount in enumerate(label_count):
            # Divided by 5 since 5 augmentations will be performed at a time
            amount_to_augment = int((goal_amount - label_amount)/5)
            while amount_to_augment*NUM_CLASSES < 5:
                amount_to_augment += 1
            print("amount to aug", amount_to_augment*NUM_CLASSES)
            print("class", label)
            tups_to_augment = random.choices(sorted_tup[label], k=amount_to_augment)

            to_augment_x = [(tups_to_augment[0])[0]]
            to_augment_y = np.empty(amount_to_augment)
            to_augment_y.fill(label)

            x_org_len = len(x_aug)

            for i in range(1, amount_to_augment):
                to_augment_x = np.append(to_augment_x, [(tups_to_augment[i])[0]], axis=0)

            for i in range(0, amount_to_augment): #amount to augment
                i = 0
                # for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1,
                #             save_to_dir ='preview/' + str(label),
                #             save_prefix = str(label) + '_image', save_format ='png'):
                for batch in datagen.flow(to_augment_x, to_augment_y, batch_size=1):
                    x_aug = np.append(x_aug[:], batch[0], axis=0)
                    y_aug = np.append(y_aug[:], batch[1], axis=0)
                    # print("{}/{}".format((len(x_aug) - x_org_len), amount_to_augment*NUM_CLASSES))
                    i += 1
                    if i > 5:
                        break

        norm_done = time.time()
        print("Augmenting normalization finished after {0} seconds".format(norm_done - start_aug))
        label_count = [0] * NUM_CLASSES
        for lab in y_aug:
            label_count[int(lab)] += 1
        print('label count after norm:', label_count)

    if SAVE_AUGMENTATION_TO_HDF5:
        start_sav = time.time()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # file path for the created .hdf5 file
        hdf5_path = dir_path + '/Messidor2_PNG_AUG_' + str(IMG_HEIGHT) + '.hdf5'
        print(hdf5_path)

        # open a hdf5 file and create earrays
        f = h5py.File(hdf5_path, mode='w')

        f.create_dataset("x_aug", data=x_aug, dtype='uint8')
        f.create_dataset("y_aug", data=y_aug, dtype='uint8')

        f.close()
        save_done = time.time()
        print("Saving finished after {0} seconds".format(save_done - start_sav))
    return (x_aug, y_aug)


def load_data(path, to_shuffle):
    '''' Load a dataset from a hdf5 file '''
    with h5py.File(path, "r") as f:
        (x_train_load, y_train_load, x_test_load, y_test_load) = np.array(f['x_train']), np.array(f['y_train']), np.array(f['x_test']), np.array(f['y_test'])
    train_label_count = [0] * NUM_CLASSES
    test_label_count = [0] * NUM_CLASSES
    for lab in y_train_load:
        train_label_count[lab] += 1
    for lab in y_test_load:
        test_label_count[lab] += 1

    if to_shuffle:
        (x_train_load, y_train_load) = shuffle_data(x_train_load, y_train_load)
        (x_test_load, y_test_load) = shuffle_data(x_test_load, y_test_load)

    if AUGMENTATION:
        (x_train_load, y_train_load) = data_augmentation(x_train_load, y_train_load, train_label_count)
        print("augmentation of train set done")
        train_label_count = [0] * NUM_CLASSES
        for lab in y_train_load:
            train_label_count[int(lab)] += 1
        
        if AUGMENT_TESTSET:
            (x_test_load, y_test_load) = data_augmentation(x_test_load, y_test_load, test_label_count)
            print("augmentation test set done")
            test_label_count = [0] * NUM_CLASSES
            for lab in y_test_load:
                test_label_count[int(lab)] += 1

    return (x_train_load, y_train_load), (x_test_load, y_test_load), train_label_count, test_label_count


def prepare_data():
    ''' Load the data and perform shuffle/augmentations if needed '''
    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test), train_label_count, test_label_count = load_data(DATA_PATH, TO_SHUFFLE)

    # For evaluation, this image is put in the fig_dir created above
    test_img_idx = random.randint(0, len(x_test) - 1)

    print("""dataset_name = {}, batch_size = {}, num_classes = {}, epochs_1 = {},
        epochts_2 = {}, test_img_idx = {},
        train_test_split = {}, to_shuffle = {}, augmentation = {}, train_label_count = {},
        test_label_count = {}, label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {},
        train_all_layers = {}, weights_to_use = {}, train_val_split = {}, MIN_DELTA = {}, Early_monitor = {},
        es_patience_1 = {}, es_patience_2 = {}, train_val_split = {}""".format(
            DATASET_NAME, BATCH_SIZE, NUM_CLASSES, EPOCHS_1,
            EPOCHS_2, test_img_idx,
            TRAIN_TEST_SPLIT, TO_SHUFFLE, AUGMENTATION, train_label_count,
            test_label_count, LABEL_NORMALIZER, SAVE_AUGMENTATION_TO_HDF5, LEARN_RATE,
            TRAIN_ALL_LAYERS, WEIGHTS_TO_USE, TRAIN_VAL_SPLIT, MIN_DELTA, EARLY_MONITOR,
            ES_PATIENCE_1, ES_PATIENCE_2, TRAIN_VAL_SPLIT))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES*2)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES*2)

    return(x_train, y_train, x_test, y_test, test_img_idx)


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

    # TODO: test first abs of y_pred_cat
    # TODO: abs later testing 
    # Is error only modelled after being right? Or also wrong?
    # y_pred_cat_abs = K.abs(y_pred_cat)
    y_true_var = K.square(y_pred_cat - y_true_cat)
    y_true_var_abs = K.abs(y_true_var)
    y_pred_var = y_pred[:, NUM_CLASSES:]
    var_loss = K.mean(K.square(y_pred_var - y_true_var_abs), axis=-1)
    total_loss = cat_loss + var_loss

    return total_loss


def main():
    ''' Main function '''
    # Load data
    x_train, y_train, x_test, y_test, test_img_idx = prepare_data()

    # VGG16 since it does not include batch normalization of dropout by itself
    variance_model = VGG16(weights=WEIGHTS_TO_USE, include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH),
                       classes=NUM_CLASSES)

    # Stacking a new simple convolutional network on top of vgg16
    all_layers = [l for l in variance_model.layers]
    x = all_layers[0].output
    for i in range(1, len(all_layers)):
        x = all_layers[i](x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    last_layer = Dense(4096, activation='relu', name='fc2')(x)
    classification = Dense(NUM_CLASSES, activation='softmax')(last_layer)
    variance = Dense(NUM_CLASSES, activation='linear')(last_layer)

    out = concatenate([classification, variance])

    # Creating new model
    variance_model = Model(inputs=all_layers[0].input, outputs=out)

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
    fig_dir = os.path.join(os.getcwd(), DATANAME + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(fig_dir)
    # Dir to store Tensorboard data
    log_dir = os.path.join(fig_dir, "logs" + os.path.sep + "fit" + os.path.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir)

    os.chdir(fig_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor=EARLY_MONITOR, min_delta = MIN_DELTA,
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_1)


    datagen = ImageDataGenerator(rescale=1./255, dtype='ndarray')
    train_generator = datagen.flow(x_train[0:int(TRAIN_VAL_SPLIT*len(x_train))],
                                   y_train[0:int(TRAIN_VAL_SPLIT*len(y_train))],
                                   batch_size=BATCH_SIZE)
    
    val_generator = datagen.flow(x_train[int(TRAIN_VAL_SPLIT*len(x_train)):],
                                 y_train[int(TRAIN_VAL_SPLIT*len(y_train)):],
                                 batch_size=BATCH_SIZE)
    # mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)


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

    early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor=EARLY_MONITOR, min_delta = MIN_DELTA,
                                                      mode='auto', verbose=1, patience=ES_PATIENCE_2)


    variance_model.fit(train_generator,
                       epochs=EPOCHS_2,
                       verbose=2,
                       validation_data=val_generator,
                       callbacks=[tensorboard_callback, early_stopping_2])

    # variance_model = load_model('best_model.h5')
    # os.remove('best_model.h5')

    # Save JSON config to disk
    json_config = variance_model.to_json()
    with open('variance_model_config.json', 'w') as json_file:
        json_file.write(json_config)
    # Save weights to disk
    variance_model.save_weights('variance_weights.h5')

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
    mean_var = []

    #Convert var pred to uncertainty
    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = np.abs(pred[NUM_CLASSES:])

        for i in range(0, NUM_CLASSES):
            raw_var = var[i]
            if_true_error = pow((classif[i] - y_test[ind][i]), 2)
            var[i] = abs(if_true_error - raw_var)
        variance_predictions[ind][NUM_CLASSES:] = var

    for ind, pred in enumerate(variance_predictions):
        true_label = true_labels[ind]
        classif = pred[:NUM_CLASSES]
        classif_ind = np.argmax(classif)
        var = pred[NUM_CLASSES:]
        mean_var.append(np.mean(var))

        # for i in range(0, NUM_CLASSES):
        #     raw_var = var[i]
        #     if_true_error = pow((classif[i] - y_test[ind][i]), 2)
        #     var[i] = abs(if_true_error - raw_var)

        var_pred = var[classif_ind]
        var_correct = var[true_label]
        var_low = np.argmin(var)

        if classif_ind != true_label:
            wrong += 1
            # print("Pred: {}, true: {}".format(classif_ind, true_label))
            # print("Var_pred: {}, var_true: {}".format(var_wrong, var_correct))
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
    print("Mean_var: {:.2%}, var_acc = {:.2%}".format(np.mean(mean_var), 1.0-(np.mean(mean_var))))
    print("mean mean variance:", np.mean(variance_predictions))


    confusion = tf.math.confusion_matrix(labels=y_test.argmax(axis=1), predictions=variance_predictions[:, :NUM_CLASSES].argmax(axis=1),
                                    num_classes=NUM_CLASSES)
    print(confusion)

    single_acc = accuracy_score(y_test.argmax(axis=1), variance_predictions[:, :NUM_CLASSES].argmax(axis=1))
    print(single_acc)
    dir_path_head_tail = os.path.split(os.path.dirname(os.getcwd()))
    if WEIGHTS_TO_USE != None:
        new_path = dir_path_head_tail[0] + os.path.sep + DATANAME + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + WEIGHTS_TO_USE + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    else:
        new_path = dir_path_head_tail[0] + os.path.sep + DATANAME + os.path.sep + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + str(BATCH_SIZE) + 'B' + '_{:.1%}A'.format(single_acc)
    os.rename(fig_dir, new_path)

if __name__ == "__main__":
    main()
