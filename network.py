# -*- coding: utf-8 -*-
# Builds the DNN model for the CANDELS Paper
# ----------------------------------
# Layers------------------------------------
# 45x45x3 input
# 6x6x32 Convolutional Layer: Stride: 1x1
# 5x5x64 Convolutional Layer: Stride: 1x1
# 20x20 Max Pooling Layer: Stride: 2x2
# 3x3x128 Convolution Layer: Stride: 1x1
# 8x8 Max Pooling Layer: Stride: 2x2
# 3x3x128 Convolution Layer: Stride: 1x1
# 2x2 Max Polling Lyaer: Stride: 2x2
# 2048x1 Fully Connected Layer
# 2048x1 Fully Connected Layer
# 5x1 Linear Layer
# Classes are f_sph, f_disk, f_irr, f_ps, f_unc

# Convnet setup ref:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3
# %20-%20Neural%20Networks/convolutional_network.py
from datahelper import DataHelper
import tensorflow as tf


# Network Parameters
n_input = 6075  # 45x45 6075?
n_classes = 5

# Parameters
learning_rate = 0.001  # get numbers from paper?
training_iters = 200  # get numbers from paper?
batch_size = 5  # batch size has to divide evenly into total example
display_step = 5


# MODEL HELPERS ---------------------------------------------------------------
def conv2d(img, w, b):
    # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#conv2d
    # shouldn't w be a kernel?''
    x = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
    z = tf.nn.bias_add(x, b)
    return tf.nn.relu(z)


def max_pool(img, k):
    # https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#max_pool
    kernel = [1, k, k, 1]
    stride = [1, k, k, 1]
    return tf.nn.max_pool(img, ksize=kernel, strides=stride, padding='VALID')


def conv_net(_X, _weights, _biases):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 45, 45, 3])
    print _X.get_shape()

    # First convolution layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    print conv1.get_shape()

    # Second Covolution layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    print conv2.get_shape()
    conv2 = max_pool(conv2, k=2)
    print conv2.get_shape()

    # Thrid Convolution Layer
    conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
    print conv3.get_shape()
    conv3 = max_pool(conv3, k=2)
    print conv3.get_shape()

    # Fourth Convolution Layer
    conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'])
    print conv4.get_shape()
    conv4 = max_pool(conv4, k=2)
    print conv4.get_shape()

    # First Fully Connected Layer
    fc1 = tf.reshape(conv4, [1, -1])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, _weights['wf1']), _biases['bf1']))

    # Second Fully Connected Layer
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, _weights['wf2']), _biases['bf2']))

    # Output
    output = tf.add(tf.matmul(fc2, _weights['out']), _biases['out'])
    return output

weights = {
    # 6x6 conv, 3-channel input, 32-channel outputs
    'wc1': tf.Variable(tf.truncated_normal([6, 6, 3, 32], stddev=0.01)),
    # 5x5 conv, 32-channle inputs, 64-channel outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)),
    # 3x3 conv, 64-channel inputs, 128-channel outputs
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01)),
    # 3x3 conv, 128-channel inputs, 128-channel outputs
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
    # fully connected, 9*128 inputs, 2048 outputs
    'wf1': tf.Variable(tf.truncated_normal([batch_size * 1152, 2048], stddev=0.001)),
    # fully coneected 2048 inputs, 2048 outputs
    'wf2': tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([2048, n_classes], stddev=0.01))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bc3': tf.Variable(tf.truncated_normal([128])),
    'bc4': tf.Variable(tf.truncated_normal([128])),
    'bf1': tf.Variable(tf.truncated_normal([2048])),
    'bf2': tf.Variable(tf.truncated_normal([2048])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}
# MODEL HELPERS ---------------------------------------------------------------

# INPUT Layer
x = tf.placeholder(tf.float32, [batch_size, 45, 45, 3])
y = tf.placeholder(tf.float32, [None, n_classes])

# Construct model
pred = conv_net(x, weights, biases)

# Loss and optimizer
# https://www.tensorflow.org/versions/r0.8/api_docs/python/nn.html#tanh
# Possibly change this to sigmoid?
# Use squared error, becuase our output doesn't reduce to probability dist'
cost = tf.reduce_mean(tf.squared_difference(tf.tanh(pred), y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate Model
# http://stackoverflow.com/questions/33846069
rmse = tf.sqrt(tf.reduce_mean(pred - y))

# Initialize
init = tf.initialize_all_variables()

dh = DataHelper(batch_size, test_idx=5000)

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        # TODO get data in proper format
        batch_xs, batch_ys = dh.get_next_batch()

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        if step % display_step == 0:
            acc = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            print "Iter " + str(step * batch_size) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training RMSE= " + "{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"

    print "Testing Error:"
    tst_x, tst_y = dh.get_test_data(batch_size)
    test_error = sess.run(rmse, feed_dict={x: tst_x, y: tst_y})
    print test_error