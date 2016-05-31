
from datahelper import DataHelper
import tensorflow as tf
import sys
# sess = tf.InteractiveSession() # remember to remove

# Network Parameters
IMAGE_SIZE = 45
NUM_LABELS = 5

# Parameters
learning_rate = 0.001  # get numbers from paper?
training_iters = 500  # get numbers from paper?
batch_size = 5  # batch size has to divide evenly into total example
display_step = 5

# Input place holder nodes.
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])


# Convolution Layer 1

# Weights & biases CL1
W_conv1 = tf.Variable(tf.truncated_normal([6,6,3,32], stddev=0.01))
b_conv1 = tf.Variable(tf.truncated_normal([32]))

conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

print 'ConvPool 1'
print W_conv1.get_shape()
print b_conv1.get_shape()
print conv1.get_shape()
print h_conv1.get_shape()
print pool1.get_shape()


# Convolution Layer 2
# Weights & biases CL2
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.01))
b_conv2 = tf.Variable(tf.truncated_normal([64]))

conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
h_conv2 = tf.nn.relu(conv2 + b_conv2)

pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 2'
print W_conv2.get_shape()
print b_conv2.get_shape()
print conv2.get_shape()
print h_conv2.get_shape()
print pool2.get_shape()



# Convolution Layer 3
# Weights & biases CL3
W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.01))
b_conv3 = tf.Variable(tf.truncated_normal([128]))

conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
h_conv3 = tf.nn.relu(conv3 + b_conv3)

#pool3 = tf.nn.max_pool(h_conv3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 3'
print W_conv3.get_shape()
print b_conv3.get_shape()
print conv3.get_shape()
print h_conv3.get_shape()
#print pool3.get_shape()


# Convolution Layer 4
# Weights & biases CL4
W_conv4 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.01))
b_conv4 = tf.Variable(tf.truncated_normal([128]))

conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID')
h_conv4 = tf.nn.relu(conv4 + b_conv4)

pool4 = tf.nn.max_pool(h_conv4, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 4'
print W_conv4.get_shape()
print b_conv4.get_shape()
print conv4.get_shape()
print h_conv4.get_shape()
print pool4.get_shape()


# Fully connected Layer1
W_fc1 = tf.Variable(tf.truncated_normal([2*2*128, 2048], stddev=0.001))
b_fc1 = tf.Variable(tf.truncated_normal([2048]))

pool4_flat = tf.reshape(pool4, [-1, 2*2*128])
h_fc1 = tf.nn.relu(tf.matmul(pool4_flat, W_fc1) + b_fc1)

print 'FC 1'
print W_fc1.get_shape()
print b_fc1.get_shape()
print pool4_flat.get_shape()
print h_fc1.get_shape()


# Fully connected Layer2
W_fc2 = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001))
b_fc2 = tf.Variable(tf.truncated_normal([2048]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

print 'FC 2'
print W_fc2.get_shape()
print b_fc2.get_shape()
print h_fc2.get_shape()


# Output Layer
W_out = tf.Variable(tf.truncated_normal([2048, NUM_LABELS], stddev=0.001))
b_out = tf.Variable(tf.truncated_normal([NUM_LABELS]))

out = tf.nn.relu(tf.matmul(h_fc2, W_out) + b_out)

print 'OUT'
print W_out.get_shape()
print b_out.get_shape()
print out.get_shape()


# No changes to old network.py beyond this. Will be updating this soon.


cost = tf.reduce_mean(tf.squared_difference(tf.tanh(out), y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


rmse = tf.sqrt(tf.reduce_mean(out - y_))

# Initialize
init = tf.initialize_all_variables()

dh = DataHelper(batch_size, test_idx=5000)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Looks like training iters in the number of images to process
    while step * batch_size < training_iters:
        # TODO get data in proper format
        batch_xs, batch_ys = dh.get_next_batch()
        #print batch_xs.shape, batch_ys.shape
        #sys.exit(0)
        sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})

        if step % display_step == 0:
            acc = sess.run(rmse, feed_dict={x: batch_xs, y_: batch_ys})
            loss = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})

            print "Iter " + str(step * batch_size) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training RMSE= " + "{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"

    print "Testing Error:"
    tst_x, tst_y = dh.get_test_data(batch_size)
    test_error = sess.run(rmse, feed_dict={x: tst_x, y_: tst_y})
    print test_error
