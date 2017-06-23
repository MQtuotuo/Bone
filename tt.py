import csv
from collections import defaultdict
import numpy as np
import os
import glob

columns = defaultdict(list) # each value in each column is appended to a list

with open('data/female.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

idx_all = np.array(columns['img_num'], dtype=float)
image_labels = columns['age']
image_labels = np.array(image_labels, dtype=float)
#image_labels = np.transpose(image_labels)
print(idx_all)
print(image_labels)
print(np.shape(image_labels))
print(type(image_labels))





files = sorted(glob.glob(os.path.join('data', 'female', '*.jpg')))
print(os.path.join('data/female', 'jpg', '*.jpg'))
labels = np.array(list(zip(files, image_labels)))

print(files)
print(labels)

