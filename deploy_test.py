#!/usr/bin/python

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import argparse
import papl

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--test", action="store_true", help="Run test")
argparser.add_argument("-d", "--deploy", action="store_true", help="Run deploy with seven.png")
argparser.add_argument("-m", "--model", default="./model_ckpt_dense", help="Specify a target model file")
args = argparser.parse_args()

if (args.test or args.deploy) == True:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
else:
    argparser.print_help()
    sys.exit()

# sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess = tf.InteractiveSession()
# sess = tf.Session()

def imgread(path):
    tmp = papl.imread(path)
    img = np.zeros((28,28,1))
    img[:,:,0]=tmp[:,:,0]
    return img

# Declare weight variables
dense_w={
    "w_conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="w_conv1"),
    "b_conv1": tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1"),
    "w_conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="w_conv2"),
    "b_conv2": tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2"),
    "w_fc1": tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name="w_fc1"),
    "b_fc1": tf.Variable(tf.constant(0.1, shape=[1024]), name="b_fc1"),
    "w_fc2": tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name="w_fc2"),
    "b_fc2": tf.Variable(tf.constant(0.1, shape=[10]), name="b_fc2")
}

def dense_cnn_model(weights):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, weights["w_fc2"]) + weights["b_fc2"])
    return y_conv

# Construct a dense model with variables
if args.test == True:
    x = tf.placeholder("float", shape=[None, 784])
    x_image = tf.reshape(x, [-1,28,28,1])
elif args.deploy == True:
    img = imgread("./seven.png")
    x = tf.placeholder("float", shape=[None, 28, 28, 1])
    x_image = x
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")

y_conv = dense_cnn_model(dense_w)

# Restore values of variables
saver = tf.train.Saver(dense_w)
saver.restore(sess, args.model)

# Calc results
if args.test == True:
    # Evaluate test sets
    import time
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # To avoid OOM, run validation with 500/10000 test dataset
    b = time.time()
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result += accuracy.eval(feed_dict={x: batch[0],
                                          y_: batch[1],
                                          keep_prob: 1.0})
    result /= 20
    a = time.time()

    print("test accuracy %g" % result)
    print "time: %s s" % (a-b)
elif args.deploy == True:
    # Infer a single image & check its latency
    import time

    b = time.time()
    result = sess.run(tf.argmax(y_conv,1), feed_dict={x:[img], y_:mnist.test.labels, keep_prob: 1.0})
    a = time.time()

    print "output: %s" % result
    print "time: %s s" % (a-b)
    papl.log("performance_ref.log", a-b)
