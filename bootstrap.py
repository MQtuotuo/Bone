#!/usr/bin/env python
import os
import glob
import urllib
import tarfile
import numpy as np
from scipy.io import loadmat
from shutil import copyfile, rmtree
from sklearn.cross_validation import train_test_split

import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve


import config

data_path = 'data'


def download_file(url, dest=None):
    if not dest:
        dest = os.path.join(data_path, url.split('/')[-1])
    urlretrieve(url, dest)


# Download the bone age dataset into the current directory
if not os.path.exists(data_path):
    os.mkdir(data_path)


# Get image labels from female.csv (the 'age' column, just a round now)
import csv
from collections import defaultdict
import numpy as np

columns = defaultdict(list) # each value in each column is appended to a list

with open('data/female.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k
idx_all = np.array(columns['img_num'], dtype=np.int)
image_labels = columns['age']
image_labels = np.array(image_labels, dtype=np.int)
#image_labels = np.transpose(image_labels)

#print(image_labels)
#print(np.shape(image_labels))
#print(type(image_labels))

# Read .mat file containing training, testing, and validation sets.
#setid = loadmat(setid_path)

#idx_train = setid['trnid'][0] - 1
#idx_test = setid['tstid'][0] - 1
#idx_valid = setid['valid'][0] - 1
idx_train, idx_test = train_test_split(idx_all, test_size=0.3)
idx_test, idx_valid = train_test_split(idx_test, test_size=0.5)


# Read .mat file containing image labels.
#image_labels = loadmat(image_labels_path)['labels'][0]

# Subtract one to get 0-based labels
#image_labels -= 1


files = sorted(glob.glob(os.path.join(data_path, 'female', '*.jpg')))
labels = np.array(list(zip(files, image_labels)))
#labels = np.dstack((files, image_labels))


# Get current working directory for making absolute paths to images
cwd = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(config.data_dir):
    rmtree(config.data_dir, ignore_errors=True)
os.mkdir(config.data_dir)

print('finished')


def move_files(dir_name, labels):
    cur_dir_path = os.path.join(config.data_dir, dir_name)
    if not os.path.exists(cur_dir_path):
        os.mkdir(cur_dir_path)

    for i in range(0, 19):
        class_dir = os.path.join(config.data_dir, dir_name, str(i))
        os.mkdir(class_dir)

    for label in labels:
        src = str(label[0])
        dst = os.path.join(cwd, config.data_dir, dir_name, label[1], src.split(os.sep)[-1])
        copyfile(src, dst)


#print(labels)
#print(idx_train)
#print(idx_train[0])
#print(idx_all)
#idx = idx_all.tolist().index(idx_train[0])
#print(idx)

def get_index(idx_array):
    index_array = np.array([], dtype=np.int)
    for i in idx_array:
        idx = idx_all.tolist().index(i)
        index_array = np.append(index_array, idx)
    return index_array

move_files('train', labels[get_index(idx_train), :])
move_files('test', labels[get_index(idx_test), :])
move_files('valid', labels[get_index(idx_valid), :])
