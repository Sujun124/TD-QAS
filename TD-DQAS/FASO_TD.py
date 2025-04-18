import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import datetime
# import utils
import os
import random
import pickle
import copy

import task_VQE
import DQAS
import vqe_task_paralell3
import VQE_train_0803
import argparse
import junjian


def best_architecture2presnt_DQAS(best_archi):
    # in archi_VQE_training, [0,1,1] present the operator Rx gate operate in qubit 1, [4,1,2] present XX gate operate in qubit 1 and 2
    # change architecture format form VQE_training to DQAS format like below, they are numbered from 0 to 11
    operator_pool = ["Rx-even", "Rx-odd", "Ry-even", "Ry-odd", "Rz-even", "Rz-odd",
                     "XX-even", "XX-odd", "YY-even", "YY-odd", "ZZ-even", "ZZ-odd"]
    re_list = []
    for i_archi in range(len(best_archi)):

        # for i_archi in range(len(best_archi)):
        element_list = []

        for j_archi_operator in range(10):
            gate = best_archi[i_archi][6 + 3 * j_archi_operator][0]
            if gate in [0, 1, 2, 3]:
                op = 'single'
            else:
                op = 'double'

            qubit_index = best_archi[i_archi][6 + 3 * j_archi_operator][1]
            if qubit_index == 0:
                qubit_op = 'even'
            else:
                qubit_op = 'odd'

            if op == 'single' and qubit_op == 'even':
                element = [1, 3, 5, 6, 7, 8, 9, 10, 11]
            elif op == 'single' and qubit_op == 'odd':
                element = [0, 2, 4, 6, 7, 8, 9, 10, 11]
            elif op == 'double' and qubit_op == 'even':
                element = [0, 1, 2, 3, 4, 5, 7, 9, 11]
            elif op == 'double' and qubit_op == 'odd':
                element = [0, 1, 2, 3, 4, 5, 7, 9, 11]
            element_list.append(element)
        re_list.append(element_list)
    return re_list


def gate_list_sample(param):
    """
    这个函数是专门为qas这种需要频繁切换线路设计的
    :param param: 单个门的旋转角参数
    :return: 当前门的表示方式的集合，它会被输入make_c的select_gate函数中，最终只会有一个奏效
    这个专门为了这个gate_set下，假如以后要换量子门种类，这个还是作为输入吧，这个作为一个例子放在这里
    """

    K = tc.set_backend("tensorflow")

    l = [
        tc.gates.Gate(K.eye(4)),
        tc.gates.Gate(K.kron(tc.gates.rx_gate(theta=param).tensor, K.eye(2))),
        tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),]
    return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]



def gate_list_sample_RXCNOT(param):
    """
    这个函数是专门为qas这种需要频繁切换线路设计的，具体原理不大好讲清楚
    :param param: 单个门的旋转角参数
    :return: 当前门的表示方式的集合，它会被输入make_c的select_gate函数中，最终只会有一个奏效
    这个专门为了这个gate_set下，假如以后要换量子门种类，这个还是作为输入吧，这个作为一个例子放在这里
    """

    K = tc.set_backend("tensorflow")

    l = [
        tc.gates.Gate(K.eye(4)),
        tc.gates.Gate(K.kron(tc.gates.rx_gate(theta=param).tensor, K.eye(2))),
        tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),]
    return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]


def FA_DQAS_gatewise(n, p, FA_ch):

    """
    First architecture
    1 random select architecture
    2 compute loss
    3 select best architecture
    """
    # gate-wise



    # hyper-parameter of generating
    FA_gate_set = ['Rx', "XX"]
    ops_repr = ["rx0", "rx1", "rx2", "rx3", "rx4", "rx5",
                     "xx01", "xx12", "xx23", "xx34", "xx45", "xx50"]
    edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]

    #  hyper-parameter of circuit

    # 量子比特数、门数、操作池大小:是单量子门或者双量子门，各有6种可能，12
    n, p, ch = n, p, FA_ch

    epochs = 400  # 4000
    batch = 8
    nnp_lr = 0.06
    stp = 0.1

    VQE_task = task_VQE.VQE(gate_list_sample)
    DQAS_implementation = DQAS.DQAS(VQE_task.vqef,n, p, ch)
    averaging_loss_list, searching_loss_list, stp, stp_list = DQAS_implementation.process_FA(epochs,batch,nnp_lr,stp)

    return averaging_loss_list, searching_loss_list, stp, stp_list



def FA_DQAS_layerwise(n, p, ch):

    """
    First architecture
    1 random select architecture
    2 compute loss
    3 select best architecture
    """
    # gate-wise

    # hyper-parameter of generating
    FA_gate_set = ['Rx', "XX"]
    ops_repr = ["Rx-single", "Rx-double", "XX-single", "XX-double"]
    # edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]

    #  hyper-parameter of circuit

    # 量子比特数、门数、操作池大小:是单量子门或者双量子门，各有6种可能，12

    epochs = 400  # 4000
    batch = 16
    nnp_lr = 0.06
    stp = 0.05

    VQE_task = task_VQE.VQE(gate_list_sample)
    DQAS_implementation = DQAS.DQAS(VQE_task.vqef,n, p, ch)
    averaging_loss_list, searching_loss_list, stp = DQAS_implementation.process_FA(epochs,batch,nnp_lr,stp)

    return averaging_loss_list, searching_loss_list, stp

def stp2topology(stp_FA, num_topology, p, ch):

    stp_So_DQAS = np.zeros([p, ch])

    sorted_index_list = []
    for i in range(len(stp_FA)):
        sorted_indices = sorted(range(len(stp_FA[i])), key=lambda x: stp_FA[i][x])
        sorted_index_list.append(sorted_indices[::-1])

    layer_wise = [[0,1,2,3], [4,5,6]]
    for num_i in range(num_topology):
        for i in range(len(sorted_index_list)):
            index1 = int(sorted_index_list[i][num_i]/6)  # 6 means qubit operation position, index1 for single or double gate
            index2 = sorted_index_list[i][num_i]%6  # index means qubit position
            for item in layer_wise[index1]:
                stp_So_DQAS[i][int((item)*6+index2)] = 1
