"""
Implementation of CNN in TF
Model will predict if mole images are malignant or benign
Micah Reich
"""
import tensorflow as tf
import numpy as np
import preprocessing
import pickle
from matplotlib.pyplot import imshow
from matplotlib.image import imread
from PIL import Image
from resizeimage import resizeimage

#   General information about input data
im_height = 227  #   Height of images in pixels
im_width = 227   #   Width of images in pixels
im_depth = 3     #   Number of color channels in images – RGB color images

num_classes = 6     #   Number of classes to predict

#   Hyper-paramaters
batch_size = 128
l_rate = 0.001
training_iters = 2000

X = tf.placeholder(tf.float32, shape=[None, im_height, im_width, im_depth], name='X')   #   Placeholder variable for images
Y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Y') #   Placeholder variable for labels
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

keep_prob = 0.5  #   Placeholder variable for dropout rate


def conv2d(x, W, b, s):
    """
    Convolutional layer helper function
    :param x: input
    :param W: weight matrix used as filter
    :param b: bias term
    :param s: stride
    """
    conv_2d = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    conv_2d = tf.nn.bias_add(conv_2d, b)
    return tf.nn.relu(conv_2d)


def maxpool2d(x, k, s):
    """
    Maxpool 2D layer helper function
    :param x: input
    :param k: kernel size
    :param s: stride size
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')


def fc(x, W, b):
    """
    Fully connected layer helper function
    :param x: input
    """
    out = tf.nn.bias_add(tf.matmul(x, W), b)
    return tf.nn.relu(out)


def dropout(x, keep_prob):
    """
    Dropout layer helper function
    :param x: input
    :param keepPro: dropout percentage
    """
    return tf.nn.dropout(x, keep_prob)


def conv_net(x, weights, biases, drop_rate):

    x_reshaped = tf.reshape(x, shape=[-1, im_width, im_height, im_depth])

    #   Conv2D: 96 kernels of size 11x11 with a stride of 4 and padding of 0
    conv1 = conv2d(x_reshaped, weights['W_c1'], biases['b_c1'], 4)

    #   Maxpool2D: pooling size of 3×3 and stride 2.
    conv1 = maxpool2d(conv1, 3, 2)

    #   Conv2D: 256 kernels of size 5x5 with a stride of 1 and padding of 2
    conv2 = conv2d(conv1, weights['W_c2'], biases['b_c2'], 1)

    #   Maxpool2D: pooling size of 3×3 and stride 2.
    conv2 = maxpool2d(conv2, 3, 2)

    #   Conv2D: 384 kernels of size 3x3 with a stride of 1 and padding of 1
    conv3 = conv2d(conv2, weights['W_c3'], biases['b_c3'], 1)

    #   Conv2D: 384 kernels of size 3x3 with a stride of 1 and padding of 1
    conv4 = conv2d(conv3, weights['W_c4'], biases['b_c4'], 1)

    #   Conv2D: 256 kernels of size 3x3 with a stride of 1 and padding of 1
    conv5 = conv2d(conv4, weights['W_c5'], biases['b_c5'], 1)

    #   Maxpool2D: pooling size of 3×3 and stride 2.
    conv5 = maxpool2d(conv5, 3, 2)

    #   FC1 has 4096 neurons
    flatten = tf.layers.flatten(conv5)
    fc1 = tf.contrib.layers.fully_connected(flatten, 4096)
    fc1 = dropout(fc1, keep_prob)

    fc2 = tf.contrib.layers.fully_connected(fc1, 4096)
    fc2 = dropout(fc2, keep_prob)

    out = tf.contrib.layers.fully_connected(fc2, num_classes)
    out = tf.nn.softmax(out)

    return out


weights = {
    'W_c1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'W_c2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'W_c3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'W_c4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'W_c5': tf.Variable(tf.random_normal([3, 3, 384, 256])),

    'W_fc1': tf.Variable(tf.random_normal([256*6*6, 4096])),
    'W_fc2': tf.Variable(tf.random_normal([4096, 4096])),
    'W_fc3': tf.Variable(tf.random_normal([4096, num_classes]))
}

biases = {
    'b_c1': tf.Variable(tf.random_normal([96])),
    'b_c2': tf.Variable(tf.random_normal([256])),
    'b_c3': tf.Variable(tf.random_normal([384])),
    'b_c4': tf.Variable(tf.random_normal([384])),
    'b_c5': tf.Variable(tf.random_normal([256])),

    'b_fc1': tf.Variable(tf.random_normal([4096])),
    'b_fc2': tf.Variable(tf.random_normal([4096])),
    'b_fc3': tf.Variable(tf.random_normal([num_classes]))
}

pred = conv_net(X, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
