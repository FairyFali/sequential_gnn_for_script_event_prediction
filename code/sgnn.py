# -*- coding: utf-8 -*-#
# Name:         sgnn
# Description:  Sequential GNN Model
# Author:       fali wang
# Date:         2020/1/5 17:26
import torch
from torch import nn
import torch.nn.functional as F
import math

class GNN(nn.Module):
    '''
    Graph Neual Netword
    '''
    def __init__(self, hidden_size, T):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.T = T
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_ah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.w_ih_2 = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh_2 = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih_2 = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh_2 = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_ah_2 = nn.Parameter(torch.Tensor(self.hidden_size))

        # self.dropout=nn.Dropout(dropout_p)
        self.reset_parameters()

    def GNNCell(self, A, hidden, w_ih, w_hh, b_ih, b_hh, b_ah):
        input = torch.matmul(A.transpose(1, 2), hidden) + b_ah  # A^T * h, [batch_size, 节点数量, hidden_size]
        # input=self.dropout(input)
        gi = F.linear(input, w_ih, b_ih)  # [batch_size, 节点数量, 3*hidden_size]
        gh = F.linear(hidden, w_hh, b_hh)  # [batch_size, 节点数量, 3*hidden_size]
        i_r, i_i, i_n = gi.chunk(3, 2)  # [batch_size, 节点数量, hidden_size]
        h_r, h_i, h_n = gh.chunk(3, 2)  # [batch_size, 节点数量, hidden_size]
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)  # * 对位相乘
        hy = newgate + inputgate * (hidden - newgate)
        # hy=self.dropout(hy)
        return hy

    def forward(self, A, hidden):
        # 邻接矩阵A的维度应该和hidden的行数一样
        hidden_out = hidden
        for i in range(self.T):
            hidden_in = hidden_out
            hidden_out = self.GNNCell(A, hidden_in, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
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
        self.embed_size = config.hidden_dim
        self.hidden_dim = config.hidden_dim*4
        self.batch_size = config.batch_size

        self.use_lstm = config.use_lstm

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding.weight.data.copy_(torch.from_numpy(word_vec))
        # LSTM Module
        if config.bidirectional:
            lstm_hidden_dim = config.hidden_dim*3//2
        else:
            lstm_hidden_dim = config.hidden_dim*3
        self.left_rnn = nn.LSTM(config.hidden_dim*3, lstm_hidden_dim, num_layers=config.num_layers,
                                bidirectional=config.bidirectional, dropout=config.dropout_p, batch_first=True)
        self.right_rnn = nn.LSTM(config.hidden_dim*3, lstm_hidden_dim, num_layers=config.num_layers,
                                bidirectional=config.bidirectional, dropout=config.dropout_p, batch_first=True)
        # GNN Module
        self.gnn = GNN(self.hidden_dim, config.T)
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
        # lstm
        if self.use_lstm:
            if unk_loc>0 and unk_loc<7:
                # output,(batch, seq_len, num_directions * hidden_size),
                # output.view(batch, seq_len, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively.
                left_output, _ = self.left_rnn(hidden_rnn[:, 0:unk_loc,:])
                right_output, _ = self.right_rnn(hidden_rnn[:, unk_loc:8, :])
                lstm_output = torch.cat((left_output, right_output, hidden_rnn[:,8:,:]), dim=1) # [batch_size, 13, 128*4]
            elif unk_loc==0:
                right_output, _ = self.right_rnn(hidden_rnn)
                lstm_output = right_output
            elif unk_loc==7:
                left_output, _ = self.left_rnn(hidden_rnn)
                lstm_output = left_output
            lstm_output = torch.cat((hidden0_13, lstm_output), dim=2)
        else:
            lstm_output = hidden
        # gnn
        gnn_output = self.gnn(A, lstm_output)
        # attention
        input_a = gnn_output[:, 0:8, :].repeat(1,5,1).view(5*len(gnn_output), 8, -1) # [5*batch_size, 8, 128*4]
        input_b = gnn_output[:, 8:13, :] # [batch_size, 5, 128*4]
        u_a = F.relu(self.linear_u_one(input_a))  # [5*batch_size, 8, 128*2]
        u_a2 = F.relu(self.linear_u_one2(u_a))  # [5*batch_size, 8, 1]
        u_b = F.relu(self.linear_u_two(input_b))  # [batch_size, 5, 128*2]
        u_b2 = F.relu(self.linear_u_two2(u_b))  # [batch_size, 5, 1]
        u_c = torch.add(u_a2.view(5 * len(gnn_output), 8), u_b2.view(5 * len(gnn_output), 1))  # [5*batch_size, 8]
        weight = torch.exp(F.tanh(u_c))
        weight = (weight / torch.sum(weight, 1).view(-1, 1)).view(-1, 8, 1)
        weighted_input = torch.mul(input_a, weight)  # 对位相乘
        a = torch.sum(weighted_input, 1)  # [5*batch_size, 128*4]
        b = input_b / 8.0
        b = b.view(5 * len(gnn_output), -1)  # [5*batch_size, 128*4]
        # similarity
        if metric == 'euclid':
            scores = self.metric_euclid(a, b)

        return scores


    def evaluate(self, A, input, targets, unk_loc, dev_index, metric='euclid'):
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
        num_correct = torch.sum(L[:, 0] == targets).type(torch.FloatTensor)
        batch_size = len(targets)
        accuracy = num_correct/batch_size * 100.0
        return accuracy


    def metric_euclid(self, v0, v1):
        return -torch.norm(v0-v1, 2, 1).view(-1,5)

