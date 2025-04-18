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
from DQAS import DQAS
import matplotlib.pyplot as plt
import task_VQE
import argparse


def stp2topology(stp_FA, num_topology, p, ch):
    stp_So_DQAS = np.zeros([p, ch])

    sorted_index_list = []
    for i in range(len(stp_FA)):
        sorted_indices = sorted(range(len(stp_FA[i])), key=lambda x: stp_FA[i][x])
        sorted_index_list.append(sorted_indices[::-1])

    layer_wise = [[0, 1, 2, 3], [4, 5, 6]]
    for num_i in range(num_topology):
        for i in range(len(sorted_index_list)):
            index1 = int(
                sorted_index_list[i][num_i] / 6)  # 6 means qubit operation position, index1 for single or double gate
            index2 = sorted_index_list[i][num_i] % 6  # index means qubit position
            for item in layer_wise[index1]:
                stp_So_DQAS[i][int((item) * 6 + index2)] = 1


    return stp_So_DQAS


def purely_DQAS(args=None):
    import junjian


    seed = args.seed

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    np.random.seed(seed)

    time1, _ = junjian.read_curr_time()

    n, p, ch = args.n,args.p,args.ch

    VQE_task = task_VQE.VQE()
    DQAS_implementation = DQAS(VQE_task.vqef, n, p, ch)
    averaging_loss_list, searching_loss_list, searching_circuit = DQAS_implementation.process(args.epochs,
                                                                                              args.pure_batch,
                                                                                              args.pure_nnp_lr,
                                                                                              args.pure_stp_lr)
    time2, _ = junjian.read_curr_time()
    cost_time = time2 - time1

    return averaging_loss_list, searching_loss_list, searching_circuit, cost_time


