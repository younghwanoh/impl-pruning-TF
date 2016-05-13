#!/usr/bin/python

def _complex_concat(a, b):
    tmp = []
    for i in a:
        for j in b:
            tmp.append(i+j)
    return tmp

def _add_prefix(a):
    tmp = []
    for idx, val in enumerate(a):
        tmp.append("w_" + val)
        # tmp.append("b_" + val)
    return tmp

# Pruning threshold setting (70 % off)
th = {
    "w_conv1":  0.110689 ,
    "b_conv1":  0.0830755 ,
    "w_conv2":  0.0988535 ,
    "b_conv2":  0.088152 ,
    "w_fc1":  0.0979934 ,
    "b_fc1":  0.0991785 ,
    "w_fc2":  0.105566 ,
    "b_fc2":  0.100518
}

# CNN settings for pruned training
debug = True
target_layer = ["fc1", "fc2"]

# Output data lists
target_all_layer = _add_prefix(target_layer)

target_dat = _complex_concat(target_all_layer, [".dat"])
target_p_dat = _complex_concat(target_all_layer, ["_p.dat"])
target_tp_dat = _complex_concat(target_all_layer, ["_tp.dat"])
target_stf_dat = _complex_concat(target_all_layer, ["_stf.dat"])

data_all = target_dat + target_p_dat + target_tp_dat + target_stf_dat

# Data settings
show_zero = False

# Graph settings
alpha = 0.75
step = 0.003
color = "green"
pdf_prefix = ""
