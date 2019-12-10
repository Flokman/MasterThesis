########################## first part: prepare data ###########################
from random import shuffle
import glob
import os 
import os.path
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + '/Messidor2/IMAGES'

img_rows = 256
img_cols = 256
img_depth = 3

shuffle_data = True  # shuffle the addresses

hdf5_path = dir_path + '/Messidor2_' + str(img_rows) + '.hdf5'  # file path for the created .hdf5 file
print(hdf5_path)

images_path = dir_path + '/Messidor2/IMAGES/*' # the original data path
print(images_path)

# get all the image paths 
addrs = glob.glob(images_path)
baseaddrs = []
for im in addrs:
    if im.endswith('JPG'):
        im = im[:-3]
        im = im + 'jpg'
    baseaddrs.append(str(os.path.basename(im)))

labels = []
with open(dir_path + '/Messidor2' + '/messidor2_classes.csv') as f:
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
    


# shuffle data
if shuffle_data: 
    combined = list(zip(addrs, labels)) # use zip() to bind the images and labels together
    shuffle(combined)
 
    (addrs, labels) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "addrs" then contains all the shuffled paths and 
                               # "labels" contains all the shuffled labels.
                               
# Divide the data into 80% for train and 20% for test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

##################### second part: create the h5py object #####################
import numpy as np
import h5py

train_shape = (len(train_addrs), img_rows, img_cols, img_depth)
test_shape = (len(test_addrs), img_rows, img_cols, img_depth)

# open a hdf5 file and create earrays 
f = h5py.File(hdf5_path, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("x_train", train_shape, np.uint8)
f.create_dataset("x_test", test_shape, np.uint8)  

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key. 
f.create_dataset("y_train", (len(train_addrs),), np.uint8)
f["y_train"][...] = train_labels

f.create_dataset("y_test", (len(test_addrs),), np.uint8)
f["y_test"][...] = test_labels

######################## third part: write the images #########################
import cv2

# loop over train paths
for i in range(len(train_addrs)):
  
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)) )

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)# resize to (img_rows, img_cols)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
    f["x_train"][i, ...] = img[None] 

# loop over test paths
for i in range(len(test_addrs)):

    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)) )

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["x_test"][i, ...] = img[None]

f.close()