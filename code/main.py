# -*- coding: utf-8 -*-#
# Name:         main
# Description:  
# Author:       fali wang
# Date:         2020/1/4 14:16
import utils
from utils import DataLoader
import pickle
from config import Config
import time
from sgnn import SGNN
import torch

def train(model, ans_loc, train_data, valid_data, dev_index, config, metric='euclid'):
    '''
    train the model
    :return:
    '''
    iteration_times = config.iteration_times
    batch_size = config.batch_size
    patients = config.patients

    print('start training.')

    start = time.time()
    best_acc = 0.0
    epoch = 0
    patient = 0
    while True:
        for iter in range(iteration_times):
            data, flag = train_data.next_batch(batch_size)
            if flag:
                print('ecpoch', epoch, 'finished.')
                epoch += 1
            A, input_data, targets = data
            model.train() # train mode
            scores = model(A, input_data, unk_loc=ans_loc, metric=metric)

            loss = model.loss_fn(scores, targets)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            # eval
            A, input_data, targets = valid_data.all_data()
            model.eval()
            accuracy = model.evaluate(A, input_data, targets, ans_loc, dev_index, metric=metric)
            # print
            if iter % 50 == 0:
                print("iter", iter, ', eval acc', accuracy.item(), ', metric', metric)
            # save best model.
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), '../data/sgnn_{}.model'.format(ans_loc))
                print('save model.')
                patient = 0
            else:
                patient += 1
            # judge the patient
            if patient > patients:
                break
        if iter != iteration_times-1:
            break

    print('train finished. Best acc {:.2f}, Epoch {}, Time {}'.format(best_acc, epoch, time.time()-start))
    model.eval()

    return best_acc


if __name__ == '__main__':

    config = Config()

    if config.data_type == 'origin':
        train_data = DataLoader(pickle.load(open('../data/train_8_data.data', 'rb')))
        valid_data = DataLoader(pickle.load(open('../data/valid_8_data.data', 'rb')))
        test_data = DataLoader(pickle.load(open('../data/test_8_data.data', 'rb')))
        ans_loc = 8 # original data, the index of correct answer is 7(namely 8th)
    elif 'trans' in config.data_type:
        ans_loc = int(config.data_type[-1])
        train_data = DataLoader(pickle.load(open('../data/train_{}_data.pkl'.format(ans_loc), 'rb')))
        valid_data = DataLoader(pickle.load(open('../data/valid_{}_data.pkl'.format(ans_loc), 'rb')))
        test_data = DataLoader(pickle.load(open('../data/test_{}_data.pkl'.format(ans_loc), 'rb')))
        
    print("ans_loc:{}, data_type:{}, use_lstm:{}, batch_size:{}".format(ans_loc, config.data_type, config.use_lstm, config.batch_size))
    print('train data prepare done.')
    dev_index = pickle.load(open('../data/dev_index.pickle', 'rb'))
    test_index = pickle.load(open('../data/test_index.pickle', 'rb'))
    word_vec = utils.get_word_vec('../data/deepwalk_128_unweighted_with_args.txt')
    print('word vector prepare done.')
    # define model
    model = SGNN(word_vec, config)
    model = utils.trans_to_cuda(model)
    # train model
    best_acc = train(model, ans_loc, train_data, valid_data, dev_index, config)
    # record the experiment result
    with open('best_result.txt', 'a') as f:
        f.write('Best Acc: %f, L2_penalty=%s ,MARGIN=%s ,LR=%s ,T=%s ,BATCH_SIZE=%s ,Iteration_times=%s ,PATIENTS=%s, HIDDEN_DIM=%s, METRIC=%s\n' % (best_acc, config.l2_penalty, config.margin, config.lr, config.T, config.batch_size, config.iteration_times, config.patients, config.hidden_dim, config.metric))
    f.close()




