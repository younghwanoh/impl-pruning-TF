#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import readline
import rlcompleter
if 'libedit' in readline.__doc__:
    readline.parse_and_bind("bind ^I rl_complete")
else:
    readline.parse_and_bind("tab: complete")

import tensorflow as tf
import numpy as np
import argparse
import sys
import papl

import scipy.sparse as sp

argparser = argparse.ArgumentParser()
argparser.add_argument("-1", "--first_round", action="store_true",
    help="Run 1st-round: train with 20000 iterations")
argparser.add_argument("-2", "--second_round", action="store_true",
    help="Run 2nd-round: apply pruning and its additional training")
argparser.add_argument("-3", "--third_round", action="store_true",
    help="Run 3rd-round: transform model to a sparse format and save it")
argparser.add_argument("-c", "--checkpoint", default="./model_ckpt_dense",
    help="Target checkpoint file for 2nd and 3rd round")
args = argparser.parse_args()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
if (args.first_round or args.second_round or args.third_round) == False:
    argparser.print_help()
    sys.exit()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

sess = tf.InteractiveSession()

def apply_prune(weights):
    total_fc_byte = 0
    total_fc_csr_byte = 0
    total_nnz_elem = 0
    total_origin_elem = 0

    dict_nzidx = {}

    for target in papl.config.target_layer:
        wl = "w_" + target
        print(wl + " threshold:\t" + str(papl.config.th[wl]))

        # Get target layer's weights
        weight_obj = weights[wl]
        weight_arr = weight_obj.eval()

        # Apply pruning
        weight_arr, w_nzidx, w_nnz = papl.prune_dense(weight_arr, name=wl,
                                            thresh=papl.config.th[wl])

        # Store pruned weights as tensorflow objects
        dict_nzidx[wl] = w_nzidx
        sess.run(weight_obj.assign(weight_arr))

    return dict_nzidx

def apply_prune_on_grads(grads_and_vars, dict_nzidx):
    # Mask gradients with pruned elements
    for key, nzidx in dict_nzidx.items():
        count = 0
        for grad, var in grads_and_vars:
            if var.name == key+":0":
                nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
                grads_and_vars[count] = (tf.mul(nzidx_obj, grad), var)
            count += 1
    return grads_and_vars

def gen_sparse_dict(dense_w):
    sparse_w = dense_w
    for target in papl.config.target_all_layer:
        target_arr = np.transpose(dense_w[target].eval())
        sparse_arr = papl.prune_tf_sparse(target_arr)
        sparse_w[target+"_idx"]=tf.Variable(tf.constant(sparse_arr[0],dtype=tf.int32),
            name=target+"_idx")
        sparse_w[target]=tf.Variable(tf.constant(sparse_arr[1],dtype=tf.float32),
            name=target)
        sparse_w[target+"_shape"]=tf.Variable(tf.constant(sparse_arr[2],dtype=tf.int32),
            name=target+"_shape")
    return sparse_w

dense_w={
    "w_conv1": tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name="w_conv1"),
    "b_conv1": tf.Variable(tf.constant(0.1,shape=[32]), name="b_conv1"),
    "w_conv2": tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name="w_conv2"),
    "b_conv2": tf.Variable(tf.constant(0.1,shape=[64]), name="b_conv2"),
    "w_fc1": tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1), name="w_fc1"),
    "b_fc1": tf.Variable(tf.constant(0.1,shape=[1024]), name="b_fc1"),
    "w_fc2": tf.Variable(tf.truncated_normal([1024,10],stddev=0.1), name="w_fc2"),
    "b_fc2": tf.Variable(tf.constant(0.1,shape=[10]), name="b_fc2")
}

def dense_cnn_model(weights):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, weights["w_fc2"]) + weights["b_fc2"])
    return y_conv

def test(y_infer, message="None."):
    correct_prediction = tf.equal(tf.argmax(y_infer,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    result = accuracy.eval(feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels,
                                      keep_prob: 1.0})
    print(message+" %g\n" % result)
    return result

def check_file_exists(key):
    import os
    fileList = os.listdir(".")
    count = 0
    for elem in fileList:
        if elem.find(key) >= 0:
            count += 1
    return key+"-"+str(count)

# Construct a dense model
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")

y_conv = dense_cnn_model(dense_w)

saver = tf.train.Saver()

if args.first_round == True:
    # First round: Train original model
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    score = test(y_conv, message="First-round test accuracy")
    
    # Save model objects to readable format
    papl.print_weight_vars(dense_w, papl.config.target_all_layer,
                           papl.config.target_dat, show_zero=papl.config.show_zero)
    # Save model objects to serialized format
    saver.save(sess, "./model_ckpt_dense", write_meta_graph=False)

if args.second_round == True:
    # Second round: Retrain pruned model, start with default model: model_ckpt_dense
    saver.restore(sess, args.checkpoint)

    # Apply pruning on this context
    dict_nzidx = apply_prune(dense_w)

    # save model objects to readable format
    papl.print_weight_vars(dense_w, papl.config.target_all_layer,
                           papl.config.target_p_dat, show_zero=papl.config.show_zero)

    # Test prune-only networks
    score = test(y_conv, message="Second-round prune-only test accuracy")

    # save model objects to serialized format
    saver.save(sess, "./model_ckpt_dense_pruned", write_meta_graph=False) 

    # Retrain networks
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    trainer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = trainer.compute_gradients(cross_entropy)
    grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
    train_step = trainer.apply_gradients(grads_and_vars)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize firstly touched variables (mostly from accuracy calc.)
    for var in tf.all_variables():
        if tf.is_variable_initialized(var).eval() == False:
            sess.run(tf.initialize_variables([var]))

    # Train 4 epochs additionally
    for i in range(4400):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Save retrained variables to a desne form
    key = check_file_exists("model_ckpt_dense_retrained")
    saver.save(sess, key, write_meta_graph=False)

    # Test the retrained model
    score = test(y_conv, message="Second-round final test accuracy")

if args.third_round == True:
    # Third round: Transform iteratively pruned model to a sparse format and save it
    if args.second_round == False:
        saver.restore(sess, args.checkpoint)

    # Transform final weights to a sparse form
    sparse_w = gen_sparse_dict(dense_w)

    # Initialize new variables in a sparse form
    for var in tf.all_variables():
        if tf.is_variable_initialized(var).eval() == False:
            sess.run(tf.initialize_variables([var]))

    # Save model objects to readable format
    papl.print_weight_vars(dense_w, papl.config.target_all_layer,
                           papl.config.target_tp_dat, show_zero=papl.config.show_zero)
    # Save model objects to serialized format
    final_saver = tf.train.Saver(sparse_w)
    final_saver.save(sess, "./model_ckpt_sparse_retrained", write_meta_graph=False) 
