import plistlib as plt
import pickle
import numpy as np
import FADQAS_SODQAS
import run_DQAS
import junjian
import sujun

import argparse


def purely2FASO_DQAS(param=None,seed=None):


    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=2023, help="random seed")

    parser.add_argument('-n', type=int, default=6, help="qubit")
    parser.add_argument('-p', type=int, default=36, help="gate number")
    parser.add_argument('-ch', type=int, default=42, help="operation pool layerwise")
    parser.add_argument('-FA_ch', type=int, default=12, help="operation pool layerwise FASO")


    parser.add_argument('-epochs', type=int, default=500, help="DQAS epoch")


    # pure DQAS param
    parser.add_argument('-pure_batch', type=int, default=8, help="DQAS_batch")
    parser.add_argument('-pure_stp_lr', type=int, default=0.2, help="stp_lr")
    parser.add_argument('-pure_nnp_lr', type=int, default=0.06, help="nnp_lr")

    # FASO DQAS param
    parser.add_argument('-FASO_batch', type=int, default=8, help="DQAS_batch")
    parser.add_argument('-FASO_stp_lr', type=int, default=0.5, help="stp_lr")
    parser.add_argument('-FASO_nnp_lr', type=int, default=0.05, help="nnp_lr")
    parser.add_argument('-num_topology', type=int, default=1, help="DQAS num_topology")

    parser.add_argument('-FA_init', type=int, default=0.015, help="FASO FA_init")

    parser.add_argument('-Note: layer_wise_DQAS_FASO', type=int, default=1, help="DQAS num_topology")

    args = parser.parse_args()

    args.FA_init = 1 / args.ch


    if seed is not None:
        args.seed = seed

    print("seed",seed)

    [args.pure_stp_lr, args.pure_nnp_lr, args.pure_batch] = param

    time_record1, time_save = junjian.read_curr_time()

    print("SO_DQAS")
    averaging_loss_list1, searching_loss_list1, searching_archi1, cost_time1, stp \
        = FADQAS_SODQAS.FA_SO_DQAS(args)

    averaging_loss_list1, searching_loss_list1, searching_archi1, cost_time1, stp \
        = [],[],[],[],[]

    print("purely_DQAS")
    averaging_loss_list0, searching_loss_list0, searching_circuit0, cost_time0 \
        = run_DQAS.purely_DQAS(args)

    time_record2, _ = junjian.read_curr_time()
    save_result = [["ave_lost","search_lost","searching_circuit","args_item","cost_time0"],
                   [averaging_loss_list0, searching_loss_list0, searching_circuit0, cost_time0],
                   [averaging_loss_list1, searching_loss_list1, searching_archi1, cost_time1]
                   ]

    args_item = junjian.load_argsitem(args)
    time_cost = time_record2 - time_record1

    name =  "result/exp" + str(args.seed) + '_' + str(time_save) + ".pkl"
    sujun.save_pkl([save_result,tag,time_cost, args_item, stp], name)

if __name__ == "__main__":

    purely2FASO_DQAS()


