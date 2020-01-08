# -*- coding: utf-8 -*-#
# Name:         config
# Description:  the configuration of the model.
# Author:       fali wang
# Date:         2020/1/5 17:16

class Config(object):
    data_type = 'origin'
    hidden_dim = 128
    l2_penalty = 1e-5
    lr = 1e-4
    T = 1 # T GNN cells
    margin = 0.015 # margin loss parameter
    batch_size = 1000
    iteration_times = 520
    patients = 1000
    metric = 'euclid' # evaluate the similarity between existing and condidate
    dropout_p = 0.
    # LSTM
    use_lstm = True
    num_layers = 1
    bidirectional = True
    # attention

