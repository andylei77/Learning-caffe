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


#load the model
net = caffe.Net('conv.prototxt', caffe.TEST)


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

#reshape the data blob (1, 1, 100, 100) to the new size (1, 1, 360, 480) to fit the image :
im = np.array(Image.open('cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

net.forward()

#Now net.blobs['conv'] is filled with data, and the 3 pictures inside each of the 3 neurons (net.blobs['conv'].data[0,i]) can be plotted easily.

#To save the net parameters net.params, just call :
net.save('mymodel.caffemodel')


