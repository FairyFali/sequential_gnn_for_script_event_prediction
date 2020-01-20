# -*- coding: utf-8 -*-#
# Name:         chain
# Description:  
# Author:       fali wang
# Date:         2020/1/13 19:08

import utils
from utils import DataLoader
import torch
import torch.nn as nn
import pickle
import argparse
import time
from torch.autograd import Variable
import sys

HIDDEN_DIM = 128*4
L2_penalty = 1e-8
LR = 0.0001
MARGIN = 0.015
BATCH_SIZE = 1000
ITERATION_TIMES = 600
PATIENTS = 1000
DROPOUT = 0.

# gru process method
version = 1
bidirectional_g = True

class EventChain(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec, num_layers, bidirectional):
        super(EventChain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word_vec))
        self.embedding.weight.requires_grad = False
        if version == 1:
            hidden_size = embedding_dim*2 if bidirectional else embedding_dim*4
            self.rnn = nn.GRU(embedding_dim*4, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=DROPOUT, batch_first=True)
        elif version == 2:
            hidden_size = embedding_dim*3//2 if bidirectional else embedding_dim*3
            self.rnn = nn.GRU(embedding_dim*3, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=DROPOUT, batch_first=True)
        self.bidirectioinal = bidirectional
        self.hidden_dim = hidden_dim

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        linear_input_dim = embedding_dim*4
            
        self.linear_s_one=nn.Linear(linear_input_dim, 1, bias=False)
        self.linear_s_two=nn.Linear(linear_input_dim, 1, bias=True)
        self.linear_u_one=nn.Linear(linear_input_dim, 1, bias=False)
        self.linear_u_two=nn.Linear(linear_input_dim, 1, bias=True)

        self.loss_fn = nn.MultiMarginLoss(margin=MARGIN)

        model_grad_params = filter(lambda p: p.requires_grad == True, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)

        self.optimizer = torch.optim.RMSprop([{'params': tune_params}, {'params': self.embedding.parameters(), 'lr': LR * 0.06}], lr=LR,
            weight_decay=L2_penalty, momentum=0.2)

    def compute_scores(self,output):
        # output   #5000*9*(128*4)
        a=self.linear_s_one(output[:,0:8,:])
        b=self.linear_s_two(output[:,8,:])
        c=torch.add(a.view(-1,8),b)
        scores=torch.sigmoid(c)
        # attention weight matrix
        u_a=self.linear_u_one(output[:,0:8,:])
        u_b=self.linear_u_two(output[:,8,:])
        u_c=torch.add(u_a.view(-1,8),u_b)
        weight=torch.exp(torch.tanh(u_c))
        weight=weight/torch.sum(weight,1).view(-1,1)
        scores=torch.sum(torch.mul(scores,weight),1).view(-1,5)
        return scores

    def forward(self, input, unk_loc):
        '''
        forward
        :param input: [batch_size, 52, hidden_size]
        :return:
        '''
        hidden = self.embedding(input)

        # version 1, put 128*4 into gru
        # version 2, put 128*3 into gru, and concatenate the first 128
        if version == 1:
            hidden = torch.cat((hidden[:,0:13,:],hidden[:,13:26,:],hidden[:,26:39,:],hidden[:,39:52,:]), 2)
            input_a = hidden[:, 0:8, :].repeat(1, 5, 1).view(5 * len(hidden), 8, -1)  # [5000, 8, 512]
            input_b = hidden[:, 8:13, :].contiguous().view(-1, 1, 512)
            hidden = torch.cat((input_a, input_b), 1)  # 5000*9*(128*4)
            hidden_rnn = hidden
            if unk_loc > 0 and unk_loc < 8:
                left_output, hl = self.rnn(hidden_rnn[:, 0:unk_loc, :])

                right_input = hidden_rnn[:, unk_loc:8, :]
                idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                idx = utils.trans_to_cuda(torch.LongTensor(idx))
                inverted_right_input = right_input.index_select(1, idx)
                inverted_right_output, hr = self.rnn(inverted_right_input)
                right_output = inverted_right_output.index_select(1, idx)

                if self.bidirectioinal:
                    hl_init = hl[-2, :, :]  # forward
                    hr_init = hr[-2, :, :]

                    h_init = torch.cat([torch.unsqueeze(hl_init, 0), torch.unsqueeze(hr_init, 0)], dim=0)
                else:
                    hl_init = hl[-1, :, :]
                    hr_init = hr[-1, :, :]

                    h_init = torch.unsqueeze(hl_init, 0) if unk_loc >= 4 else torch.unsqueeze(hr_init, 0)

                candidate_output, _ = self.rnn(hidden_rnn[:, 8:, :], h_init)  # [batch, 1, hidden_dim]

                rnn_output = torch.cat((left_output, right_output, candidate_output), dim=1)  # [batch_size, 9, hidden_dim]
            elif unk_loc == 0:
                right_input = hidden_rnn[:, :, :]
                idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                idx = utils.trans_to_cuda(torch.LongTensor(idx))
                inverted_right_input = right_input.index_select(1, idx)
                inverted_right_output, _ = self.rnn(inverted_right_input)
                right_output = inverted_right_output.index_select(1, idx)
                rnn_output = right_output
            elif unk_loc == 8:
                left_output, _ = self.rnn(hidden_rnn)  # init h?, default torch.zeros(shape)
                rnn_output = left_output
        elif version == 2:
            hidden0_13 = hidden[:, 0:13, :]
            hidden13_26 = hidden[:, 13:26, :]
            hidden26_39 = hidden[:, 26:39, :]
            hidden39_52 = hidden[:, 39:, :]

            hidden_p = torch.cat((hidden0_13[:, 0:8, :].repeat(1, 5, 1).view(5 * len(hidden0_13), 8, -1), hidden0_13[:, 8:13, :].contiguous().view(-1, 1, self.hidden_dim//4)), dim=1) # [5*batch_size, 9, 128]

            hidden = torch.cat((hidden13_26, hidden26_39, hidden39_52), 2) # [batch_size, 13, 128*3]
            input_a = hidden[:, 0:8, :].repeat(1, 5, 1).view(5 * len(hidden), 8, -1)  # [5000, 8, 128*3]
            input_b = hidden[:, 8:13, :].contiguous().view(-1, 1, (self.hidden_dim//4)*3)
            hidden = torch.cat((input_a, input_b), 1)  # 5000*9*(128*3)
            hidden_rnn = hidden

            if unk_loc > 0 and unk_loc < 8:
                left_output, hl = self.rnn(hidden_rnn[:, 0:unk_loc, :])

                right_input = hidden_rnn[:, unk_loc:8, :]
                idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                idx = utils.trans_to_cuda(torch.LongTensor(idx))
                inverted_right_input = right_input.index_select(1, idx)
                inverted_right_output, hr = self.rnn(inverted_right_input)
                right_output = inverted_right_output.index_select(1, idx)

                if self.bidirectioinal:
                    hl_init = hl[-2, :, :]  # forward
                    hr_init = hr[-2, :, :]

                    h_init = torch.cat([torch.unsqueeze(hl_init, 0), torch.unsqueeze(hr_init, 0)], dim=0)
                else:
                    hl_init = hl[-1, :, :]
                    hr_init = hr[-1, :, :]

                    h_init = torch.unsqueeze(hl_init, 0) if unk_loc >= 4 else torch.unsqueeze(hr_init, 0)

                candidate_output, _ = self.rnn(hidden_rnn[:, 8:, :], h_init)  # [batch, 1, hidden_dim]

                rnn_output = torch.cat((left_output, right_output, candidate_output), dim=1)  # [batch_size, 9, hidden_dim]
            elif unk_loc == 0:
                right_input = hidden_rnn[:, :, :]
                idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                idx = utils.trans_to_cuda(torch.LongTensor(idx))
                inverted_right_input = right_input.index_select(1, idx)
                inverted_right_output, _ = self.rnn(inverted_right_input)
                right_output = inverted_right_output.index_select(1, idx)
                rnn_output = right_output
            elif unk_loc == 8:
                left_output, _ = self.rnn(hidden_rnn)  # init h?, default torch.zeros(shape)
                rnn_output = left_output
            rnn_output = torch.cat((hidden_p, rnn_output), dim=2)
        scores=self.compute_scores(rnn_output)
        return scores

    def evaluate(self, input, targets, unk_loc, dest=sys.stdout):
        with torch.no_grad():
            scores = self.forward(input, unk_loc)
        _, L = torch.sort(scores,descending=True)
        if dest != sys.stdout:
            pred = L[:, 0].tolist()
            y = targets.tolist()
            pickle.dump((pred, y), dest)
        num_correct = torch.sum((L[:,0] == targets).type(torch.FloatTensor))
        samples = len(targets)
        accuracy = num_correct / samples *100.0
        return accuracy


def train(model, ans_loc, train_data, valid_data):
    acc_list = []
    best_acc = 0.
    print('starting train.')
    start = time.time()

    epoch = 0
    patient = 0
    while True:
        for iter in range(ITERATION_TIMES):
            data, flag = train_data.next_batch(BATCH_SIZE)
            if flag:
                print('ecpoch', epoch, 'finished.')
                epoch += 1
            A, input_data, targets = data
            model.train()
            scores = model(input_data, unk_loc=ans_loc)

            loss = model.loss_fn(scores, targets)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            # eval
            A, input_data, targets = valid_data.all_data()
            model.eval()
            accuracy = model.evaluate(input_data, targets, ans_loc)
            acc_list.append((time.time()-start, accuracy.item()))
            # print
            if iter % 50 == 0:
                print("iter", iter, ', eval acc', accuracy.item())

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), '../data/event_chain_{}.model'.format(ans_loc))
                print('save model.')
                patient = 0
            else:
                patient += 1
            # judge the patient
            if patient > PATIENTS:
                break
        if iter != ITERATION_TIMES-1:
            break

    print('train finished. Best acc {:.2f}, Epoch {}, Time {}'.format(best_acc, epoch, time.time()-start))
    pickle.dump(acc_list, open('../data/event_chain_{}_acc_list.pickle'.format(ans_loc), 'wb'))
    model.eval()

    return best_acc

def evaluate(model_file, data, ans_loc, save_results=False):
    '''
    evaluate the data by the model file.
    :param model_file:
    :param data:
    :return:
    '''
    word_vec = utils.get_word_vec('../data/deepwalk_128_unweighted_with_args.txt')
    model = utils.trans_to_cuda(EventChain(embedding_dim=HIDDEN_DIM//4, hidden_dim=HIDDEN_DIM, vocab_size=len(word_vec), word_vec=word_vec, num_layers=1, bidirectional=bidirectional_g))
    model.load_state_dict(torch.load(model_file))

    if save_results:
        filename = model_file[:model_file.rindex('.')] + '_result.pkl'
        dest = open('../data/'+filename, 'wb')
    else:
        dest = sys.stdout

    model.eval()
    A, input_data, targets = data.all_data()
    accuracy = model.evaluate(input_data, targets, ans_loc, dest=dest)

    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.description = 'please input the event chain arguments.'
    parser.add_argument("-l", "--ans_loc", help="this is parameter of answer location", dest="ans_loc", type=int, default="8")
    parser.add_argument("-m", "--mode", help="train or test", dest="mode", type=str, default='train')

    args = parser.parse_args()
    ans_loc = args.ans_loc
    mode = args.mode
    
    print('ans_loc:{}, mode:{}'.format(ans_loc, mode))
    
    if ans_loc >= 8:
        train_data = DataLoader(pickle.load(open('../data/train_8_data.data', 'rb')))
        valid_data = DataLoader(pickle.load(open('../data/valid_8_data.data', 'rb')))
        test_data = DataLoader(pickle.load(open('../data/test_8_data.data', 'rb')))
    else:
        train_data = DataLoader(pickle.load(open('../data/train_{}_data.pkl'.format(ans_loc), 'rb')))
        valid_data = DataLoader(pickle.load(open('../data/valid_{}_data.pkl'.format(ans_loc), 'rb')))
        test_data = DataLoader(pickle.load(open('../data/test_{}_data.pkl'.format(ans_loc), 'rb')))
    print('data prepare done.')

    if mode == 'train':
        word_vec = utils.get_word_vec('../data/deepwalk_128_unweighted_with_args.txt')
        print('word vector prepare done.')

        # define model
        model = utils.trans_to_cuda(EventChain(embedding_dim=HIDDEN_DIM//4, hidden_dim=HIDDEN_DIM, vocab_size=len(word_vec), word_vec=word_vec, num_layers=1, bidirectional=bidirectional_g))
        best_acc = train(model, ans_loc, train_data, valid_data)
    elif mode == 'test':
        accuracy = evaluate('../data/event_chain_{}.model'.format(ans_loc), test_data, ans_loc, save_results=True)
        print('best test dataset acc {:.2f}'.format(accuracy))


