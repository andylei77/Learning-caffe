#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
import caffe


DATA_ROOT = 'examples/kaggle_digit_recognizer/data'
MODEL_ROOT = 'examples/kaggle_digit_recognizer'
join = os.path.join
TEST = join(DATA_ROOT, 'test.csv')
OUTPUT = join(DATA_ROOT, 'result.csv')
CAFFE_MODEL = join(MODEL_ROOT, 'lenet_iter_10000.caffemodel')
CAFFE_SOLVER = join(MODEL_ROOT, 'lenet.prototxt')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# load test dataset
logger.info('Load test dataset from %s', TEST)
df = pd.read_csv(TEST)
data = df.values

data = data.reshape((len(data), 28, 28, 1))
data = data / 255.

# set caffe net
net = caffe.Classifier(CAFFE_SOLVER, CAFFE_MODEL)

# predict
logger.info('Start predict')
BATCH_SIZE = 100
iter_k = 0
labels = []
while True:
    logger.info('ITER %d', iter_k)
    batch = data[iter_k*BATCH_SIZE: (iter_k+1)*BATCH_SIZE]
    if batch.size == 0:
        break
    result = net.predict(batch)
    for label in np.argmax(result, 1):
        labels.append(label)
    iter_k = iter_k + 1
logger.info('Prediction Done')

# write to file
logger.info('Save result to %s', OUTPUT)
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
with open(OUTPUT, 'w') as fd:
    fd.write('ImageId,Label\n')
    for idx, label in enumerate(labels):
        fd.write(str(idx+1))
        fd.write(',')
        fd.write(str(label))
        fd.write('\n')