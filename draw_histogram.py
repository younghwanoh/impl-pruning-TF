#!/usr/bin/python

import sys
sys.dont_write_bytecode = True

import papl
import config

papl.draw_histogram(config.weight_all, step=0.01)
papl.draw_histogram(config.syn_all, step=1)
