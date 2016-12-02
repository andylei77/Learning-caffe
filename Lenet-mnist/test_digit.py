import os
import sys
import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/home/andy/caffe/'

sys.path.insert(0, caffe_root + 'python')
import caffe
MODEL_FILE = '/home/andy/caffe/examples/mnist/lenet.prototxt'
PRETRAINED = '/home/andy/caffe/examples/mnist/lenet_iter_10000.caffemodel'
IMAGE_FILE = '/home/andy/caffe/examples/images/test4.bmp'

input_image = caffe.io.load_image(IMAGE_FILE, color=False)
net = caffe.Classifier(MODEL_FILE, PRETRAINED) 
prediction = net.predict([input_image], oversample = False)
caffe.set_mode_cpu()
print 'predicted class:', prediction[0].argmax()