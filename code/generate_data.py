# -*- coding: utf-8 -*-#
# Name:         generate_data
# Description:  This is a separate file. To generate the data we need.
# Author:       fali wang
# Date:         2020/1/6 08:56
import torch
import numpy as np

def trans_to_mid_data(data, ind = 4):
    '''
    Turn the raw data into the data we need.
    :param data:
    :param ind:
    :return:
    '''
    A, input_data, targets = data
    size = len(targets)
    new_input_data = []
    new_A = []
    for i in range(size):
        target = targets[i]
        inp = input_data[i]
        a = np.copy(A[i].numpy())
        # 把ind对应的位置的数据提取，在input_data里删除掉，将最后一个正确答案删除掉，将ind的数据插入到候选答案的位置上
        ind_sample = inp
        start = 0
        t = []
        while start < len(inp):
            end = start + 13
            sub_sample = ind_sample[start:end].tolist()
            # print(sub_sample, start, end, ind, target+8)
            sub_sample[ind], sub_sample[target + 8] = sub_sample[target + 8], sub_sample[ind]
            sub_sample.insert(8, sub_sample[ind])
            del sub_sample[ind]
            t.extend(sub_sample)
            start += 13
        new_input_data.append(t)

        # a[[1,2], :] = a[[2,1], :] numpy交换两行
        a[[ind, target + 8], :] = a[[target + 8, ind], :]
        a = np.insert(a, 8, a[ind, :], axis=0)
        a = np.delete(a, [ind], axis=0)

        a[:, [ind, target + 8]] = a[:, [target + 8, ind]]
        a = np.insert(a, 8, a[:, ind], axis=1)
        a = np.delete(a, [ind], axis=1)

        new_A.append(a)

    return (torch.tensor(new_A), torch.tensor(new_input_data), targets)

