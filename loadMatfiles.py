"""
This file is to let the tensorflow model to use mat files (Which generally get read the figures by Matlab)
"""

import numpy as np
import scipy.io as sio

# the path of mat file, the "r" before the path is essential
load_fn = r'D:\dataset\SVHN\testdata.mat'

# apply sio.loadmat() function to load mat file
load_data = sio.loadmat(load_fn)
# better to print this data to get each type
# print(load_data)
# get the data
data11 = load_data['data']
# let the type of data as float32, which is essential in tensorflow
data = data11.astype(np.float32)
# get the labels
labels = load_data['labels']
# However, the type of "labels" in mat file are offen ndarray, which is quite different 
# with the classical ways. And that of the CIFAR dataset with py is list. 
# apply labels[0] to get the the label array
a = labels[0]
# apply a.tolist() function to turn ndarray to list
label = a.tolist()
