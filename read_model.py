#!/usr/bin/python

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import papl
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", required=True, help="Specify serialized input model")
argparser.add_argument("-r", "--ratio", help="Specify ratio")
args = argparser.parse_args()

def read_model_obj_with_sorted_ratio(fname, ratio):
    saver = tf.train.Saver()
    saver.restore(sess, fname)

    print str(ratio*100)+" %"
    target_obj_list = [weights[elem] for elem in papl.config.target_all_layer]
    for elem in target_obj_list:
        arr = elem.eval()
        arr = list(arr.reshape(arr.size))
        arr.sort(cmp=lambda x,y:cmp(abs(x), abs(y)))

        print "\""+elem.name[:-2]+"\": ", abs(arr[int(len(arr)*ratio)-1]), ","

def print_raw_matrix(fname):
    saver = tf.train.Saver()
    saver.restore(sess, fname)
    import numpy as np
    np.save("w_fc1.raw", weights["w_fc1"].eval())
    np.save("w_fc2.raw", weights["w_fc2"].eval())

def read_model_obj(fname):
    saver = tf.train.Saver()

    import os.path
    try:
        assert os.path.isfile(fname)
        saver.restore(sess, fname)
        switcher = {
            "model_ckpt_dense": papl.config.target_dat,
            "model_ckpt_dense_pruned": papl.config.target_p_dat,
            "model_ckpt_dense_retrained": papl.config.target_tp_dat
        }
        papl.print_weight_vars(weights, papl.config.target_all_layer, switcher.get(args.model))
    except AssertionError:
        print "Warning: No such files or directory\n"
        pass
    except:
        import sys
        print "Unexpected error:", sys.exc_info()[0]

weights = {
    "w_conv1": tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="w_conv1"),
    "b_conv1": tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1"),
    "w_conv2": tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="w_conv2"),
    "b_conv2": tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2"),
    "w_fc1": tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name="w_fc1"),
    "b_fc1": tf.Variable(tf.constant(0.1, shape=[1024]), name="b_fc1"),
    "w_fc2": tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name="w_fc2"),
    "b_fc2": tf.Variable(tf.constant(0.1, shape=[10]), name="b_fc2")
}

sess = tf.InteractiveSession()

if __name__ == "__main__":
    if bool(args.ratio) == False:
        read_model_obj(args.model)
    else:
        read_model_obj_with_sorted_ratio(args.model, float(args.ratio))
    # print_raw_matrix(args.model)
