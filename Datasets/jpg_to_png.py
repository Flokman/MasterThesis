########################## first part: prepare data ###########################
from random import shuffle
import glob
import os 
import os.path
import csv
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = dir_path + '/Messidor2/IMAGES'

img_rows = 128
img_cols = 128
img_depth = 3

shuffle_data = True  # shuffle the addresses

hdf5_path = dir_path + '/Messidor2_' + str(img_rows) + '.hdf5'  # file path for the created .hdf5 file
print(hdf5_path)

images_path = dir_path + '/Messidor2/IMAGES/*' # the original data path
print(images_path)

# get all the image paths 
addrs = glob.glob(images_path)
count = 0
for im in addrs:
    if im.endswith('JPG'):
        IM = Image.open(im)
        newname = im[:-3] + 'png'
        IM.save(newname,'png')
        count += 1

print("{} images converted to png".format(count))