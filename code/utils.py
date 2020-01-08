# -*- coding: utf-8 -*-#
# Name:         utils
# Description:  
# Author:       fali wang
# Date:         2020/1/4 14:19
import torch
import numpy as np

use_cuda = torch.cuda.is_available()

def trans_to_cuda(variable):
    '''
    transform variable to cuda device variable
    :param variable: tensor
    :return:
    '''
    if use_cuda:
        torch.cuda.set_device(3)
        return variable.cuda()
    else:
        return variable

    # return variable

def id_to_vec(embed_file):
    '''
    read word embedding from file
    :param embed_file:
    :return: dict
    '''
    dic = {} # id:embedding, 128 dim
    for s in open(embed_file):
        s = s.strip().split()
        if len(s) == 2:
            # skip the fist line
            continue
        dic[int(s[0])] = np.array(s[1:], dtype=np.float32)
    # the key 0 embedding is equal to all 0
    dic[0] = np.zeros(len(dic[0]), dtype=np.float32)
    return dic

def word_to_id(voc_file):
    '''
    voc file is id:verb, id is same to above embedd
    :param voc_file: verb/word is mapped to id
    :return:
    '''
    dic = {} # word:id
    rev_dic = {} # id:word
    for s in open(voc_file):
        s = s.strip().split()
        dic[s[1]] = int(s[0])
        rev_dic[int(s[0])] = s[1]
    return dic, rev_dic

def get_word_vec(embed_file):
    '''
    read word_vec from embed_file
    :param embed_file:
    :return:
    '''
    dic = id_to_vec(embed_file)
    size = len(dic)
    word_vec = []
    for i in range(size):
        word_vec.append(dic[i])
    return np.array(word_vec, dtype=np.float32)


class DataLoader(object):

    def __init__(self, questions):
        '''
        questions is (A, input_data, targets) 3-tuple
        :param questions:
        '''
        super(DataLoader, self).__init__()
        self.A, self.input_data, self.targets = questions # tensor

        self.corpus_length = len(self.targets)
        # pointer
        self.start = 0

    def next_batch(self, batch_size):
        start = self.start
        end = (start + batch_size) if start+batch_size<self.corpus_length else self.corpus_length
        self.start += batch_size
        if self.start < self.corpus_length:
            flag = False # There are also data.
        else:
            flag = True
            self.start = 0
        return [trans_to_cuda(self.A[start:end]), trans_to_cuda(self.input_data[start:end]), trans_to_cuda(self.targets[start:end])], flag

    def all_data(self):
        return [trans_to_cuda(self.A), trans_to_cuda(self.input_data), trans_to_cuda(self.targets)]

