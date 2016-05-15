#!/usr/bin/python

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import sys
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--test", action="store_true", help="Run test")
argparser.add_argument("-d", "--deploy", action="store_true", help="Run deploy with seven.png")
argparser.add_argument("-m", "--model", default="./model_ckpt_sparse_retrained", help="Specify a target model file")
args = argparser.parse_args()

if (args.test) == True:
    print "Error: TensorFlow 0.8 doesn't support broadcasts on sparse operations, cannot run test set now"
    sys.exit()
elif (args.deploy) == True:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
else:
    argparser.print_help()
    sys.exit()

sess = tf.InteractiveSession()
# sess = tf.Session()

def imgread(path):
    tmp = imresize(imread(path), (28,28))
    img = np.zeros((28,28,1))
    img[:,:,0]=tmp[:,:,0]
    return img

# Declare weight variables
sparse_w={
    "w_conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="w_conv1"),
    "b_conv1": tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1"),
    "w_conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="w_conv2"),
    "b_conv2": tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2"),
    "w_fc1":      tf.Variable(tf.zeros([321130],  dtype=tf.float32),name="w_fc1"),
    "w_fc1_idx":  tf.Variable(tf.zeros([321130,2],dtype=tf.int32),  name="w_fc1_idx"),
    "w_fc1_shape":tf.Variable(tf.zeros([2],     dtype=tf.int32),  name="w_fc1_shape"),
    "b_fc1":      tf.Variable(tf.zeros([1024], dtype=tf.float32), name="b_fc1"),
    "w_fc2":      tf.Variable(tf.zeros([1024],  dtype=tf.float32),name="w_fc2"),
    "w_fc2_idx":  tf.Variable(tf.zeros([1024,2],dtype=tf.int32),  name="w_fc2_idx"),
    "w_fc2_shape":tf.Variable(tf.zeros([2],     dtype=tf.int32),  name="w_fc2_shape"),
    "b_fc2":      tf.Variable(tf.zeros([10], dtype=tf.float32), name="b_fc2"),
}

def sparse_cnn_model(weights):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.squeeze(tf.reshape(h_pool2, [-1, 7*7*64]))
    h_fc1 = tf.nn.relu(tf.nn.embedding_lookup_sparse(h_pool2_flat, weights["w_fc1_ids"], weights["w_fc1"], combiner="sum") + weights["b_fc1"])
    h_fc1_drop = tf.squeeze(tf.nn.dropout(h_fc1, keep_prob))
    y_conv = tf.nn.relu(tf.nn.embedding_lookup_sparse(h_fc1_drop, weights["w_fc2_ids"], weights["w_fc2"], combiner="sum") + weights["b_fc2"])
    y_conv = tf.nn.softmax(tf.reshape(y_conv, [1, -1]))

    return y_conv

# Restore values of variables
saver = tf.train.Saver()
saver.restore(sess, args.model)

# Retrieve SparseTensor from serialized dense variables
sparse_w["w_fc1"] = tf.SparseTensor(sparse_w["w_fc1_idx"].eval(),
                                    sparse_w["w_fc1"].eval(),
                                    sparse_w["w_fc1_shape"].eval())
sparse_w["w_fc2"] = tf.SparseTensor(sparse_w["w_fc2_idx"].eval(),
                                    sparse_w["w_fc2"].eval(),
                                    sparse_w["w_fc2_shape"].eval())
sparse_w["w_fc1_ids"] = tf.SparseTensor(sparse_w["w_fc1_idx"].eval(),
                                    sparse_w["w_fc1_idx"].eval()[:,1],
                                    sparse_w["w_fc1_shape"].eval())
sparse_w["w_fc2_ids"] = tf.SparseTensor(sparse_w["w_fc2_idx"].eval(),
                                    sparse_w["w_fc2_idx"].eval()[:,1],
                                    sparse_w["w_fc2_shape"].eval())

# Construct a sparse model with retrieved variables
if args.test == True:
    x = tf.placeholder("float", shape=[None, 784])
    x_image = tf.reshape(x, [-1,28,28,1])
elif args.deploy == True:
    img = imgread("./seven.png")
    x = tf.placeholder("float", shape=[None, 28, 28, 1])
    x_image = x
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")

y_conv = sparse_cnn_model(sparse_w)

# Calc results
if args.test == True:
    # Evaluate test sets
    import time
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    b = time.time()
    result = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
    a = time.time()
    print("test accuracy %g" % result)
    print "time: %s s" % (a-b)

elif args.deploy == True: 
    # Infer a single image
    import time
    b = time.time()
    result = sess.run(tf.argmax(y_conv,1), feed_dict={x:[img], y_:mnist.test.labels, keep_prob: 1.0})
    a = time.time()
    print "output: %s" % result
    print "time: %s s" % (a-b)
