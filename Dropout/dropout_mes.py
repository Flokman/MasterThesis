from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input, layers, models, utils
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import os, datetime, time
import h5py
from random import seed, randint, shuffle
import multiprocessing


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

# Input image dimensions
img_rows, img_cols, img_depth = 256,  256, 3
dataset_name = '/Messidor2_PNG_AUG_' + str(img_rows) + '.hdf5'

batch_size = 64
num_classes = 5
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
add_dropout = True
MCDO = True
learn_rate = 0.0001

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
        start_aug = time.time()
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

            for i in range(0, amount_to_augment): #amount to augment
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
    
        norm_done = time.time()
        print("Augmenting normalization finished after {0} seconds".format(norm_done - start_aug))
        label_count = [0] * num_classes
        for lab in y:
            label_count[int(lab)] += 1
        print('label count after norm:', label_count)

    if save_augmentation_to_hdf5 == True:
        start_sav = time.time()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hdf5_path = dir_path + '/Messidor2_PNG_AUG_' + str(img_rows) + '.hdf5'  # file path for the created .hdf5 file
        print(hdf5_path)

        # open a hdf5 file and create earrays 
        f = h5py.File(hdf5_path, mode='w')

        f.create_dataset("x", data = x, dtype = 'uint8')
        f.create_dataset("y", data = y, dtype = 'uint8')    

        f.close()
        save_done = time.time()
        print("Saving finished after {0} seconds".format(save_done - start_sav))
    return (x, y)

def load_data(path, train_test_split, data_augmentation, to_shuffle):
    with h5py.File(data_path, "r") as f:
        (x, y) = np.array(f['x']), np.array(f['y'])
    label_count = [0] * num_classes
    for lab in y:
        label_count[lab] += 1

    if to_shuffle == True:
        (x, y) = shuffle_data(x, y)

    if augmentation == True:
        (x, y) = data_augmentation(x, y, label_count, label_normalizer)
        print("augmentation done")
        label_count = [0] * num_classes
        for lab in y:
            label_count[lab] += 1


    # Divide the data into a train and test set
    x_train = x[0:int(train_test_split*len(x))]
    y_train = y[0:int(train_test_split*len(y))]

    x_test = x[int(train_test_split*len(x)):]
    y_test = y[int(train_test_split*len(y)):]

    return (x_train, y_train), (x_test, y_test), label_count

# Split the data between train and test sets
(x_train, y_train), (x_test, y_test), label_count  = load_data(data_path, train_test_split, data_augmentation, to_shuffle)

test_img_idx =  randint(0, len(x_test)) # For evaluation, this image is put in the fig_dir created above

print("dataset_name = {}, batch_size = {}, num_classes = {}, epochs = {}, MCDO_amount_of_predictions = {}, MCDO_batch_size = {}, test_img_idx = {}, train_test_split = {}, to_shuffle = {}, augmentation = {}, label_count = {}, label_normalizer = {}, save_augmentation_to_hdf5 = {}, learn rate = {}".format(dataset_name, batch_size, num_classes, epochs, MCDO_amount_of_predictions, MCDO_batch_size, test_img_idx, train_test_split, to_shuffle, augmentation, label_count, label_normalizer, save_augmentation_to_hdf5, learn_rate))

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


def get_dropout(input_tensor, p = 0.5, MCDO = False):
    if MCDO:
        return Dropout(p)(input_tensor, training = True)
    else:
        return Dropout(p)(input_tensor)


# VGG16 since it does not include batch normalization of dropout by itself
def VGG16(include_top = True,
          weights = None,
          input_tensor = None,
          input_shape = None,
          pooling = None,
          classes = num_classes,
          add_dropout = True,
          MCDO = False):
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


    p_list = [0.2, 0.25, 0.3, 0.35, 0.4]
    P_i = 0

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
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    if add_dropout:
        x = get_dropout(x, p = p_list[P_i], MCDO = MCDO)
        P_i += 1


    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    if add_dropout:
        x = get_dropout(x, p = p_list[P_i], MCDO = MCDO)
        P_i += 1

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    if add_dropout:
        x = get_dropout(x, p = p_list[P_i], MCDO = MCDO)
        P_i += 1

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    if add_dropout:
        x = get_dropout(x, p = p_list[P_i], MCDO = MCDO)
        P_i += 1

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if add_dropout:
        x = get_dropout(x, p = p_list[P_i], MCDO = MCDO)
        P_i += 1

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
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
        if K.backend() == 'theano':
            utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model

# mcdo_model = keras.models.Sequential()
# mcdo_model.add(VGG16(include_top = False, weights = 'imagenet', input_shape = (img_rows, img_cols, img_depth), add_dropout = True, MCDO = True))
# mcdo_model.add(Flatten())
# mcdo_model.add(Dense(num_classes))

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

# save model and architecture to single file
mcdo_model.save("mcdo_model.h5")
print("Saved mcdo_model to disk")