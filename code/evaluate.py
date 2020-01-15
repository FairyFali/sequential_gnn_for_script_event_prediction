# -*- coding: utf-8 -*-#
# Name:         evaluate
# Description:  in according to the model file, evaluating the data
# Author:       fali wang
# Date:         2020/1/7 16:42
import utils
from sgnn import SGNN
from config import Config
import torch
import pickle
from utils import DataLoader

def evaluate(model_file, data, index, ans_loc, config):
    '''
    evaluate the data by the model file.
    :param model_file:
    :param data:
    :return:
    '''
    word_vec = utils.get_word_vec('../data/deepwalk_128_unweighted_with_args.txt')
    model = utils.trans_to_cuda(SGNN(word_vec, config))
    model.load_state_dict(torch.load(model_file))

    model.eval()
    A, input_data, targets = data.all_data()
    accuracy = model.evaluate(A, input_data, targets, ans_loc, index, metric=config.metric)

    return accuracy

if __name__ == '__main__':
    config = Config()

    if config.data_type == 'origin':
        test_data = DataLoader(pickle.load(open('../data/test_8_data.data', 'rb')))
        ans_loc = 8 # original data, the index of correct answer is 7(namely 8th)
    elif 'trans' in config.data_type:
        ans_loc = int(config.data_type[-1])
        test_data = DataLoader(pickle.load(open('../data/test_{}_data.pkl'.format(ans_loc), 'rb')))
    print("ans_loc:{}, data_type:{}, use_lstm:{}, batch_size:{}".format(ans_loc, config.data_type, config.use_lstm, config.batch_size))
    
    test_index = pickle.load(open('../data/test_index.pickle', 'rb'))
    accuracy = evaluate('../data/sgnn_{}.model'.format(ans_loc), test_data, test_index, ans_loc, config)
    print('best test dataset acc {:.2f}'.format(accuracy))
