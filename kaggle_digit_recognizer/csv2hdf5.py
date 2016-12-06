#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import h5py


DATA_ROOT = 'data'
join = os.path.join
TRAIN = join(DATA_ROOT, 'train.csv')
train_file = join(DATA_ROOT, 'mnist_train.h5')
test_file = join(DATA_ROOT, 'mnist_test.h5')

# logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# load data from train.csv
logger.info('Load data from %s', TRAIN)
df = pd.read_csv(TRAIN)
data = df.values

logger.info('Get %d Rows in dataset', len(data))

# random shuffle
np.random.shuffle(data)

# all dataset
labels = data[:, 0]
images = data[:, 1:]

# process data
images = images.reshape((len(images), 1, 28, 28))
images = images / 255.

# train dataset number
trainset = len(labels) * 3 / 4

# train dataset
labels_train = labels[:trainset]
images_train = images[:trainset]
# test dataset
labels_test = labels[trainset:]
images_test = images[trainset:]

# write to hdf5
if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(test_file):
    os.remove(test_file)

logger.info('Write train dataset to %s', train_file)
with h5py.File(train_file, 'w') as f:
    f['label'] = labels_train.astype(np.float32)
    f['data'] = images_train.astype(np.float32)

logger.info('Write test dataset to %s', test_file)
with h5py.File(test_file, 'w') as f:
    f['label'] = labels_test.astype(np.float32)
    f['data'] = images_test.astype(np.float32)

logger.info('Done')