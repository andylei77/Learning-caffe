#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

#set cpu
caffe.set_mode_cpu()
#set gpu
#caffe.set_device(0)
#caffe.set_mode_gpu()

'''
output = (input - kernel_size) / stride + 1

net.blobs for input data and its propagation in the layers :
net.blobs['data'] contains input data, an array of shape (1, 1, 100, 100) 
net.blobs['conv'] contains computed data in layer ‘conv’ (1, 3, 96, 96)
initialiazed with zeros.
print the infos: [(k, v.data.shape) for k, v in net.blobs.items()]


net.params a vector of blobs for weight and bias parameters
net.params['conv'][0] contains the weight parameters, an array of shape (3, 1, 5, 5) 
net.params['conv'][1] contains the bias parameters, an array of shape (3,)
initialiazed with ‘weight_filler’ and ‘bias_filler’ algorithms.
print the infos : [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]

print net.blobs['conv'].data.shape

'''

#load the model
net = caffe.Net('bvlc_reference_caffenet/deploy.prototxt', 'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
out = net.forward()

# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print out['prob'].argmax()

#print predicted labels
labels = np.loadtxt("ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]



