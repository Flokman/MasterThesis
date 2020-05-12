########################## first part: prepare data ###########################
from random import shuffle
import glob
import os 
import csv


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + '/Messidor2/IMAGES'

img_rows = 256
img_cols = 256
img_depth = 3

shuffle_data = True  # shuffle the addresses

hdf5_path = dir_path + '/Messidor2_PNG_' + str(img_rows) + '.hdf5'  # file path for the created .hdf5 file
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
label_count = [0] * 5
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

# shuffle data
if shuffle_data: 
    combined = list(zip(addrs, labels)) # use zip() to bind the images and labels together
    shuffle(combined)
 
    (addrs, labels) = zip(*combined)  # *combined is used to separate all the tuples in the list combined,  
                               # "addrs" then contains all the shuffled paths and 
                               # "labels" contains all the shuffled labels.
                               

##################### second part: create the h5py object #####################
import numpy as np
import h5py

addrs_shape = (len(addrs), img_rows, img_cols, img_depth)

# open a hdf5 file and create earrays 
f = h5py.File(hdf5_path, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("x", addrs_shape, np.uint8)
# f.create_dataset("x", addrs_shape, np.int32)

# the ".create_dataset" object is like a dictionary, the "labels" is the key. 
f.create_dataset("y", (len(addrs),), np.uint8)
# f.create_dataset("y", (len(addrs),), np.int32)
f["y"][...] = labels


######################## third part: write the images #########################
import cv2

def insert_in_center(noise, img):
    # h, w, d = img.shape
    # print(h,w)

    # hh, ww, dd = noise.shape
    # print(hh,ww)

    # # compute xoff and yoff for placement of upper left corner of resized image   
    # yoff = round((hh-h)/2)
    # xoff = round((ww-w)/2)
    # print(yoff,xoff)

    # # use numpy indexing to place the resized image in the center of background image
    # result = noise.copy()
    # result[yoff:yoff+h, xoff:xoff+w, :] = img
    x1 = int(.5 * noise.shape[0]) - int(.5 * img.shape[0])
    y1 = int(.5 * noise.shape[1]) - int(.5 * img.shape[1])
    x2 = int(.5 * noise.shape[0]) + int(.5 * img.shape[0])
    y2 = int(.5 * noise.shape[1]) + int(.5 * img.shape[1])
    # print("")
    # print(img.shape)
    # print(x1, y1, x2, y2)
    # print(x2-x1, y2-y1)

    if img.shape[0] != noise.shape[0]:
        while x2-x1 < img.shape[0]:
            x1 -= 1
        while x2-x1 > img.shape[0]:
            x1 += 1
    else:
        x2 = img.shape[0]


    if img.shape[1] != noise.shape[1]:
        while y2-y1 < img.shape[1]:
            y1 -= 1
        while y2-y1 > img.shape[1]:
            y1 += 1
    else:
        y2 = img.shape[1]
    
    # print("")
    # print(x1, y1, x2, y2)
    # print(x2-x1, y2-y1)
    # pasting the cropped image over the original image, guided by the transparency mask of cropped image
    # noise = Image.fromarray(noise)
    # img = Image.fromarray(img)
    # result = noise.paste(img, box=(x1, y1, x2, y2), mask=img)
    # result = np.array(result)
    
    # print(img.shape)
    noise[x1:x2 , y1:y2] = img
    # result = cv2.resize(noise, dsize=(TARGETSIZE, TARGETSIZE), interpolation=cv2.INTER_CUBIC)
    # print(result.shape)

    return noise


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# loop over train paths
for i in range(len(addrs)):
    gaussian_noise = np.zeros((img_rows, img_cols, img_depth),dtype=np.uint8)
    # gaussian_noise_R = cv2.randn(gaussian_noise, img_cols, 66)
    # gaussian_noise_G = cv2.randn(gaussian_noise, img_cols, 65)
    # gaussian_noise_B = cv2.randn(gaussian_noise, img_cols, 64)
    # gaussian_noise = np.stack((gaussian_noise_R, gaussian_noise_G, gaussian_noise_B), axis=2)    
  
    if i % 1000 == 0 and i > 1:
        print ('Image data: {}/{}'.format(i, len(addrs)) )

    addr = addrs[i]
    img = cv2.imread(addr)
    (h, w) = img.shape[:2]
    # print(img.shape)
    if h >= w:
        img = image_resize(img, height = img_cols)# resize to (img_rows, img_cols)
        print(img.shape)
    else:
        img = image_resize(img, width = img_cols)# resize to (img_rows, img_cols)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
    img = insert_in_center(gaussian_noise, img)
    # print(img.shape)
    f["x"][i, ...] = img[None] 

f.close()