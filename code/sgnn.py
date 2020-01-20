# -*- coding: utf-8 -*-#
# Name:         sgnn
# Description:  Sequential GNN Model
# Author:       fali wang
# Date:         2020/1/5 17:26
import torch
from torch import nn
import torch.nn.functional as F
import math
import utils
import sys
import pickle

class GNN(nn.Module):
    '''
    Graph Neual Netword
    '''
    def __init__(self, hidden_size, T, unit_type):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.T = T
        self.unit_type = unit_type

        if unit_type == 'gru':
            self.b_ah = nn.Parameter(torch.Tensor(hidden_size))
            self.w_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_wz = nn.Parameter(torch.Tensor(hidden_size))
            self.u_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_uz = nn.Parameter(torch.Tensor(hidden_size))
            self.w_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_wr = nn.Parameter(torch.Tensor(hidden_size))
            self.u_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_ur = nn.Parameter(torch.Tensor(hidden_size))
            self.w = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_w = nn.Parameter(torch.Tensor(hidden_size))
            self.u = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_u = nn.Parameter(torch.Tensor(hidden_size))
        elif unit_type == 'lstm':
            self.b_ah = nn.Parameter(torch.Tensor(hidden_size))
            # forget gate
            self.w_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.u_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_f = nn.Parameter(torch.Tensor(hidden_size))
            # input gate
            self.w_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.u_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_i = nn.Parameter(torch.Tensor(hidden_size))
            # output gate
            self.w_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.u_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_o = nn.Parameter(torch.Tensor(hidden_size))
            # cell candidation
            self.w_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.u_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def GNNCell(self, A, hidden):
        if self.unit_type == 'gru':
            a = torch.matmul(A.transpose(1, 2), hidden) + self.b_ah  # A^T * h, [batch_size, 节点数量, hidden_size]
            z = torch.sigmoid(F.linear(a, self.w_z, self.b_wz) + F.linear(hidden, self.u_z, self.b_uz))  # [batch_size, 节点数量, hidden_size]
            r = torch.sigmoid(F.linear(a, self.w_r, self.b_wr) + F.linear(hidden, self.u_r, self.b_ur))  # [batch_size, 节点数量, hidden_size]
            c = torch.tanh(F.linear(a, self.w, self.b_w) + F.linear(r*hidden, self.u, self.b_u))  # [batch_size, 节点数量, hidden_size]
            h_t = (1-z)*hidden + z*c  # [batch_size, 节点数量, hidden_size]
        elif self.unit_type == 'lstm':
            a = torch.matmul(A.transpose(1, 2), hidden) + self.b_ah
            f = torch.sigmoid(F.linear(a, self.w_f) + F.linear(hidden, self.u_f) + self.b_f)
            i = torch.sigmoid(F.linear(a, self.w_i) + F.linear(hidden, self.u_i) + self.b_i)
            o = torch.sigmoid(F.linear(a, self.w_o) + F.linear(hidden, self.u_f) + self.b_o)
            c_cand = torch.tanh(F.linear(a, self.w_c) + F.linear(hidden, self.u_c) + self.b_c)
            c = f*a + i*c_cand
            h_t = o * torch.tanh(c)

        return h_t

    def forward(self, A, hidden):
        # 邻接矩阵A的维度应该和hidden的行数一样
        hidden_out = hidden
        for i in range(self.T):
            hidden_in = hidden_out
            hidden_out = self.GNNCell(A, hidden_in)
        return hidden_out

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class SGNN(nn.Module):
    def __init__(self, word_vec, config):
        '''
        Initializer.
        :param word_vec: np.ndarray
        :param config: model parameters
        '''
        super(SGNN, self).__init__()
        self.vocab_size = len(word_vec)
        self.embed_size = config.hidden_dim # word embedding dimension
        self.hidden_dim = config.hidden_dim*4 # GNN input dimension
        self.batch_size = config.batch_size
        self.reverse = config.reverse
        self.bidirectioinal = config.bidirectional

        self.use_lstm = config.use_lstm
        self.use_attention = config.use_attention
        self.left_to_right = config.left_to_right

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding.weight.data.copy_(torch.from_numpy(word_vec))

        # GNN Module
        self.gnn = GNN(self.hidden_dim, config.T, config.unit_type)
        # LSTM Module
        if config.bidirectional:
            lstm_hidden_dim = config.hidden_dim*3//2
        else:
            lstm_hidden_dim = config.hidden_dim*3
        self.rnn = nn.LSTM(config.hidden_dim * 3, lstm_hidden_dim, num_layers=1,
                           bidirectional=config.bidirectional, dropout=config.dropout_p, batch_first=True)

        # Attention Module
        attention_input_dim = self.hidden_dim
        ## two fc layers
        self.linear_u_one=nn.Linear(attention_input_dim, int(0.5*attention_input_dim), bias=True)
        self.linear_u_one2=nn.Linear(int(0.5*attention_input_dim), 1, bias=True)
        ## two fc layers
        self.linear_u_two=nn.Linear(attention_input_dim, int(0.5*attention_input_dim), bias=True)
        self.linear_u_two2=nn.Linear(int(0.5*attention_input_dim), 1, bias=True)

        # loss function
        self.loss_fn = nn.MultiMarginLoss(margin=config.margin)
        # optimizer
        model_grad_params = filter(lambda p: p.requires_grad == True, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        self.optimizer = torch.optim.RMSprop([{'params':tune_params},{'params':self.embedding.parameters(),'lr':config.lr*0.06}],lr=config.lr, weight_decay=config.l2_penalty,momentum=0.2)


    def forward(self, A, input, unk_loc=4, metric='euclid'):
        '''
        forward propagation.
        :return:
        '''
        hidden = self.embedding(input) # [batch_size, 13*4, 128]
        hidden0_13 = hidden[:, 0:13, :]
        hidden13_26 = hidden[:, 13:26, :]
        hidden26_39 = hidden[:, 26:39, :]
        hidden39_52 = hidden[:, 39:, :]

        hidden = torch.cat((hidden0_13, hidden13_26, hidden26_39, hidden39_52), dim=2) # [batch_size, 13, 128*4]
        hidden_rnn = torch.cat((hidden13_26, hidden26_39, hidden39_52), dim=2) # [batch_size, 13, 128*3]
        # hidden_rnn = hidden # [batch_size, 13, 128*4]
        # lstm
        if self.use_lstm:
            if unk_loc>0 and unk_loc<8:
                left_output, (hl, cl) = self.rnn(hidden_rnn[:, 0:unk_loc, :])
                if self.reverse:
                    right_input = hidden_rnn[:, unk_loc:8, :]
                    idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                    idx = utils.trans_to_cuda(torch.LongTensor(idx))
                    inverted_right_input = right_input.index_select(1, idx)
                    if self.left_to_right:
                        inverted_right_output, (hr, cr) = self.rnn(inverted_right_input, (hl, cl))
                    else:
                        inverted_right_output, (hr, cr) = self.rnn(inverted_right_input)
                    right_output = inverted_right_output.index_select(1, idx)
                else:
                    if self.left_to_right:
                        right_output, (hr, cr) = self.rnn(hidden_rnn[:, unk_loc:8, :], (hl, cl))
                    else:
                        right_output, (hr, cr) = self.rnn(hidden_rnn[:, unk_loc:8, :])

                if self.bidirectioinal:
                    hl_init = hl[-2, :, :] # forward
                    hr_init = hr[-2, :, :] if self.reverse else hr[-1, :, :]
                    cl_init = cl[-2, :, :] # forward
                    cr_init = cr[-2, :, :] if self.reverse else cr[-1, :, :]

                    h_init = torch.cat([torch.unsqueeze(hl_init, 0), torch.unsqueeze(hr_init, 0)], dim=0)
                    c_init = torch.cat([torch.unsqueeze(cl_init, 0), torch.unsqueeze(cr_init, 0)], dim=0)
                else:
                    hl_init = hl[-1, :, :]
                    hr_init = hr[-1, :, :]
                    cl_init = cl[-1, :, :]
                    cr_init = cr[-1, :, :]

                    h_init = torch.unsqueeze(hl_init, 0) if unk_loc>=4 else torch.unsqueeze(hr_init, 0)
                    c_init = torch.unsqueeze(cl_init, 0) if unk_loc>=4 else torch.unsqueeze(cr_init, 0)

                candidate_output, _ = self.rnn(hidden_rnn[:, 8:, :], (h_init, c_init)) # [batch, 5, hidden_dim]

                lstm_output = torch.cat((left_output, right_output, candidate_output), dim=1) # [batch_size, 13, hidden_dim]
            elif unk_loc==0:
                if self.reverse:
                    right_input = hidden_rnn[:, :, :]
                    idx = [i for i in range(right_input.size(1) - 1, -1, -1)]
                    idx = utils.trans_to_cuda(torch.LongTensor(idx))
                    inverted_right_input = right_input.index_select(1, idx)
                    inverted_right_output, _ = self.rnn(inverted_right_input)
                    right_output = inverted_right_output.index_select(1, idx)
                else:
                    right_output, _ = self.rnn(hidden_rnn) # reverse
                lstm_output = right_output
            elif unk_loc==8:
                left_output, _ = self.rnn(hidden_rnn) # init h?, default torch.zeros(shape)
                lstm_output = left_output
            lstm_output = torch.cat((hidden0_13, lstm_output), dim=2)
        else:
            lstm_output = hidden
        # gnn
        gnn_output = self.gnn(A, lstm_output)
        # attention
        input_a = gnn_output[:, 0:8, :].repeat(1,5,1).view(5*len(gnn_output), 8, -1) # [5*batch_size, 8, 128*4]
        input_b = gnn_output[:, 8:13, :] # [batch_size, 5, 128*4]
        if self.use_attention:
            u_a = F.relu(self.linear_u_one(input_a))  # [5*batch_size, 8, 128*2]
            u_a2 = F.relu(self.linear_u_one2(u_a))  # [5*batch_size, 8, 1]
            u_b = F.relu(self.linear_u_two(input_b))  # [batch_size, 5, 128*2]
            u_b2 = F.relu(self.linear_u_two2(u_b))  # [batch_size, 5, 1]
            u_c = torch.add(u_a2.view(5 * len(gnn_output), 8), u_b2.view(5 * len(gnn_output), 1))  # [5*batch_size, 8]
            weight = torch.exp(torch.tanh(u_c))
            weight = (weight / torch.sum(weight, 1).view(-1, 1)).view(-1, 8, 1) # [5*batch_size, 8, 1]
            # print(weight)
        else:
            weight = utils.trans_to_cuda(torch.ones((5*len(gnn_output), 8), requires_grad=False))
            weight = (weight / torch.sum(weight, 1).view(-1, 1)).view(-1, 8, 1)  # [5*batch_size, 8, 1]
        weighted_input = torch.mul(input_a, weight)  # 对位相乘
        a = torch.sum(weighted_input, 1)  # [5*batch_size, 128*4]
        b = input_b / 8.0
        b = b.view(5 * len(gnn_output), -1)  # [5*batch_size, 128*4]
        # similarity
        if metric == 'euclid':
            scores = self.metric_euclid(a, b)

        return scores


    def evaluate(self, A, input, targets, unk_loc, dev_index, metric='euclid', dest = sys.stdout):
        '''
        Calculate model accuracy.
        :return: accuracy
        '''
        with torch.no_grad():
            scores = self.forward(A, input, unk_loc=unk_loc, metric=metric)
        if dev_index != None:
            for index in dev_index:
                scores[index] = -100.0
        _, L = torch.sort(scores, descending=True)
        if dest != sys.stdout:
            pred = L[:, 0].tolist()
            y = targets.tolist()
            pickle.dump((pred, y), dest)
            # dest.close()
        num_correct = torch.sum(L[:, 0] == targets).type(torch.FloatTensor)
        batch_size = len(targets)
        accuracy = num_correct/batch_size * 100.0
        return accuracy


    def metric_euclid(self, v0, v1):
        return -torch.norm(v0-v1, 2, 1).view(-1,5) # input, p='fro', dim=None, keepdim=False, out=None, dtype=None

