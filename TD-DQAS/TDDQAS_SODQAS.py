# import FASO_DQAS_0802
# import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import datetime

import os
import random
import pickle
import copy
import junjian
import FASO_FA
from DQAS import DQAS
import matplotlib.pyplot as plt
import task_VQE


def stp2topology(stp_FA, num_topology, p, ch):

    stp_So_DQAS = np.zeros([p, ch])

    sorted_index_list = []
    for i in range(len(stp_FA)):
        sorted_indices = sorted(range(len(stp_FA[i])), key=lambda x: stp_FA[i][x])
        sorted_index_list.append(sorted_indices[::-1])

    layer_wise = [[0,1,2,3], [4,5,6]]
    for num_i in range(num_topology):
        for i in range(len(sorted_index_list)):
            index1 = int(sorted_index_list[i][num_i]%2)  # 6 means qubit operation position, index1 for single or double gate
            index2 = sorted_index_list[i][num_i]%6  # index means qubit position
            if index2 >= 6:
                index2 = index2-6
            for item in layer_wise[index1]:
                stp_So_DQAS[i][int((item)*6+index2)] = 1

    return stp_So_DQAS


def stp2topology_layerwise(stp_FA, num_topology, p, ch):

    stp_So_DQAS = np.zeros([p, ch])

    sorted_index_list = []
    for i in range(len(stp_FA)):
        sorted_indices = sorted(range(len(stp_FA[i])), key=lambda x: stp_FA[i][x])
        sorted_index_list.append(sorted_indices[::-1])

    layer_wise = [[0,1,2,3], [4,5,6]]
    for num_i in range(num_topology):
        for i in range(len(sorted_index_list)):
            index1 = int(sorted_index_list[i][num_i]/2)  # 2 means layerwise, index1 for single or double gate
            index2 = sorted_index_list[i][num_i]%2  # index means qubit double or single
            for item in layer_wise[index1]:
                stp_So_DQAS[i][int((item)*2+index2)] = 1

    return stp_So_DQAS


def FA_SO_DQAS(args):

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.manual_seed(seed)
    np.random.seed(seed)

    n, p, ch = args.n, args.p, args.ch
    FA_ch = args.FA_ch


    time1, _ = junjian.read_curr_time()


    averaging_loss_list_fa, searching_loss_list_fa, stp_FA, stp_list = FASO_FA.FA_DQAS_gatewise(n, p, FA_ch)


    topology = stp2topology(stp_FA, args.num_topology, p, ch)
    VQE_task = task_VQE.VQE()
    DQAS_implementation = DQAS(VQE_task.vqef, n, p, ch)
    averaging_loss_list, searching_loss_list, searching_archi = DQAS_implementation.process_SO_space_pruning(
        args.epochs, args.FASO_batch, args.FASO_nnp_lr, args.FASO_stp_lr, topology,args.FA_init)

    time2, _ = junjian.read_curr_time()
    cost_time = time2 - time1

    return averaging_loss_list, searching_loss_list, searching_archi, cost_time, topology


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=0, help="random seed")
    parser.add_argument('-batch', type=int, default=16, help="DQAS_batch")
    parser.add_argument('-stp_lr', type=int, default=0.1, help="stp_lr")
    parser.add_argument('-nnp_lr', type=int, default=0.1, help="nnp_lr")
    parser.add_argument('-epochs', type=int, default=300, help="DQAS epoch")
    parser.add_argument('-FA_init', type=int, default=0.015, help="FA_init")
    parser.add_argument('-num_topology', type=int, default=1, help="DQAS num_topology")

    args = parser.parse_args()
    # test_single_function()
    FA_SO_DQAS(args)

    # if i == 146:
    #     a = "debug here"
