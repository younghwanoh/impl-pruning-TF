#!/usr/bin/python

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import argparse
import papl
import config

argparser = argparse.ArgumentParser()
argparser.add_argument("-t", "--test", action="store_true", help="Run test")
argparser.add_argument("-d", "--deploy", action="store_true", help="Run deploy with seven.png")
argparser.add_argument("-s", "--print_syn", action="store_true", help="Print synapses to .syn")
argparser.add_argument("-m", "--model", default="./model_ckpt_dense", help="Specify a target model file")
args = argparser.parse_args()

if (args.test or args.deploy or args.print_syn) == True:
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
    img = np.reshape(img, img.size)
    return img

# Restore values of variables
saver = tf.train.import_meta_graph(args.model+'.meta')
saver.restore(sess, args.model)

# Calc results
if args.test == True:
    # Evaluate test sets
    import time
    accuracy = tf.get_collection("accuracy")[0]

    # To avoid OOM, run validation with 500/10000 test dataset
    b = time.time()
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
        result += sess.run(accuracy, feed_dict={"x:0": batch[0],
                                                "y_:0": batch[1],
                                                "keep_prob:0": 1.0})
    result /= 20
    a = time.time()

    print("Test accuracy %g" % result)
    print "Time: %s s" % (a-b)
elif args.deploy == True:
    # Infer a single image & check its latency
    import time
    img = imgread('seven.png')
    y_conv = tf.get_collection("y_conv")[0]

    b = time.time()
    result = sess.run(tf.argmax(y_conv,1), feed_dict={"x:0":[img],
                                                      "y_:0":mnist.test.labels,
                                                      "keep_prob:0": 1.0})
    a = time.time()

    print "Output: %s" % result
    print "Time: %s s" % (a-b)
    papl.log("performance_ref.log", a-b)

elif args.print_syn == True:
    # Print synapses (Input data of each neuron)
    img = imgread('seven.png')
    target_syn = config.syn_all
    synapses = [ tf.get_collection(elem.split(".")[0])[0] for elem in target_syn ]
    for i,j in zip(synapses, config.syn_all):
        syn = sess.run(i, feed_dict={"x:0":[img],
                                     "y_:0":mnist.test.labels,
                                     "keep_prob:0": 1.0})
        papl.print_weight_nps(syn, j)
    print "Done! Synapse data is printed to x.syn"
