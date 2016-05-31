# Local libararies
from datahelper import DataHelper
from network import CandleNet

# Imports
import tensorflow as tf


# params
epochs = 2
batch_size = 5
test_size = 100
display_step = 5
test_split = 1000
n_classes = 5
learning_rate = 0.001  # get numbers from paper?

# input, label placeholders
x = tf.placeholder(tf.float32, [batch_size, 45, 45, 3])
y = tf.placeholder(tf.float32, [None, n_classes])

# create network
net = CandleNet.get_network(x)

# loss and optimizer
# Use squared error, becuase our output doesn't reduce to probability dist
cost = tf.reduce_mean(tf.squared_difference(net, y))
# https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate, the paper uses RMSE
# http://stackoverflow.com/questions/33846069
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(net, y)))

# variable initializer
init = tf.initialize_all_variables()

# model saver
saver = tf.train.Saver()

# train, test, and save model
with tf.Session() as sess:
    sess.run(init)

    # train
    epoch = 1
    while epoch <= epochs:
        print 'Training Epoch {}...'.format(epoch)
        # get data, test_idx = 19000 is ~83% train test split
        dh = DataHelper(batch_size, test_idx=test_split)
        # test data
        test_data, test_labels = dh.get_test_data(test_size)

        step = 1
        while step * batch_size < test_split:
            batch_xs, batch_ys = dh.get_next_batch()

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            if step % display_step == 0:
                acc = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

                print "Iter " + str(step * batch_size) + \
                      ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training RMSE= " + "{:.5f}".format(acc)

            step += 1

        print 'Epoch {} finished'.format(epoch)
        print 'Testing...'
        # test
        test_step = 1
        test_rmse = 0.0
        while test_step * batch_size < test_size:
            start = (test_step - 1) * batch_size
            end = test_step * batch_size
            batch_xs = test_data[start:end]
            batch_ys = test_labels[start:end]

            test_rmse += sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})

            test_step += 1

        test_rmse = test_rmse / test_step
        print 'Average Test RMSE:{}'.format(test_rmse)

        #save
        # TODO implement saving model
        # https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html

        epoch += 1