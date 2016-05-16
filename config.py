#!/usr/bin/python
import thspace as ths

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

# Pruning threshold setting (90 % off)
th = ths.th90

# CNN settings for pruned training
target_layer = ["fc1", "fc2"]
retrain_iterations = 4400

# Output data lists: do not change this
target_all_layer = _add_prefix(target_layer)

target_dat = _complex_concat(target_all_layer, [".dat"])
target_p_dat = _complex_concat(target_all_layer, ["_p.dat"])
target_tp_dat = _complex_concat(target_all_layer, ["_tp.dat"])

data_all = target_dat + target_p_dat + target_tp_dat

# Data settings
show_zero = False

# Graph settings
alpha = 0.75
step = 0.003
color = "green"
pdf_prefix = ""
