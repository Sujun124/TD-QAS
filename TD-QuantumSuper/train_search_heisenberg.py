import argparse
import pennylane as qml
from pennylane import numpy as np
import time
import os
import json
from model import CircuitSearchModel
# from evolution.evolution_sampler import EvolutionSampler
import junjian
import qiskit.providers.aer.noise as noise
import itertools


import numpy as np1

parser = argparse.ArgumentParser("QAS")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=200, help='num of warmup epochs')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--noise', action='store_true', default=True, help='use noise')
# parser.add_argument('--device', type=str, default='ibm_osaka', help='which device to use', choices=['default', 'ibmq-sim', 'ibmq','ibm_osaka'])
parser.add_argument('--device', type=str, default='N', help='which device to use', choices=['default', 'ibmq-sim', 'ibmq','ibm_osaka'])

parser.add_argument('--save_path', type=str, default='BeH2/noise/pure/', help='experiment name')


# circuit
parser.add_argument('--n_qubits', type=int, default=5, help='number of qubits')
parser.add_argument('--n_encode_layers', type=int, default=1, help='number of encoder layers')
parser.add_argument('--n_layers', type=int, default=8, help='number of layers')
parser.add_argument('--mutil_param_init', type=int, default=5, help='number of layers')


# QAS
parser.add_argument('--n_experts', type=int, default=5, help='number of experts')
parser.add_argument('--n_search', type=int, default=500, help='number of search')
parser.add_argument('--searcher', type=str, default='random', help='searcher type', choices=['random', 'evolution'])
parser.add_argument('--ea_pop_size', type=int, default=25, help='population size of evolutionary algorithm')
parser.add_argument('--ea_gens', type=int, default=20, help='generation number of evolutionary algorithm')


args = parser.parse_args()

time_record1, time_save = junjian.read_curr_time()

# args.save_path = args.save_path + '/' + time_save


# search setting

def heisenberg_model(n_qubits):
    coeffs = []
    ops = []

    # 添加 XX、YY 和 ZZ 项
    for i in range(n_qubits-1):
        coeffs.append(1.0)
        ops.append(qml.operation.Tensor(qml.PauliX(i), qml.PauliX(i + 1)))
        coeffs.append(1.0)
        ops.append(qml.operation.Tensor(qml.PauliY(i), qml.PauliY(i + 1)))
        coeffs.append(1.0)
        ops.append(qml.operation.Tensor(qml.PauliZ(i), qml.PauliZ(i + 1)))

        coeffs.append(1.0)
        ops.append(qml.PauliZ(i))

    # 周期边界条件：添加最后一对（n-1, 0）
    coeffs.append(1.0)
    ops.append(qml.operation.Tensor(qml.PauliX(n_qubits - 1), qml.PauliX(0)))
    coeffs.append(1.0)
    ops.append(qml.operation.Tensor(qml.PauliY(n_qubits - 1), qml.PauliY(0)))
    coeffs.append(1.0)
    ops.append(qml.operation.Tensor(qml.PauliZ(n_qubits - 1), qml.PauliZ(0)))

    coeffs.append(1.0)
    ops.append(qml.PauliZ(4))

    hamiltonian = qml.Hamiltonian(coeffs, ops)
    return hamiltonian



if args.noise or args.device in ['ibmq-sim', 'ibmq']:
    import qiskit
    import qiskit.providers.aer.noise as noise


def expert_evaluator(model, subnet, n_experts, cost_fn):
    r''' In this function, we locate the expert that achieves the minimum loss, where such an expert is the best choice
     for the given subset'''
    target_expert = 0
    target_loss = None
    for i in range(n_experts):
        model.params = model.get_params(subnet, i)
        temp_loss = cost_fn(model.params)
        if target_loss is None or temp_loss < target_loss:
            target_loss = temp_loss
            target_expert = i
    return target_expert




def search_topology(model, subnet):
    pass



def main(seed=None, param=None):
    if seed:
        args.seed = seed

    if param:
        args.n_layers = param

    np.random.seed(args.seed)

    com_quantum_resource = []

    save_path = args.save_path + '/' + f'n_experts{args.n_experts}' + '/' + f'n_layers{args.n_layers}_3' + '/' + f'seed{args.seed}'
    junjian.creat_file(save_path)

    # 设置量子比特数量
    n_qubits = 5

    # 获取横场伊辛模型的哈密顿量
    hamiltonian = heisenberg_model(n_qubits)
    args.n_qubits = n_qubits

    exact_value = -8.472135954999562

    wires = [i for i in range(args.n_qubits)]
    # from scipy.sparse.linalg import eigsh  # 用于稀疏矩阵的特征值计算
    #
    # H_sparse = hamiltonian.sparse_matrix(wire_order=range(n_qubits))
    # eigenvalues, eigenvectors = eigsh(H_sparse, k=1, which='SA')
    # exact_value = eigenvalues[0]
    # print(exact_value)
    # exit()

    '''init device'''
    com_quantum_resource = []

    if args.device in ['ibmq-sim', 'ibmq', 'ibm_osaka']:
        from qiskit import IBMQ
        account_key = 'XX'
        assert account_key != '', 'You must fill in your IBMQ account key.'
        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)
        if args.device == 'ibmq':
            dev = qml.device('qiskit.ibmq', wires=4, backend='ibmq_ourense', provider=provider)
        else:
            # ibm_osaka
            backend = provider.get_backend('ibm_osaka')
            # backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=args.n_qubits, noise_model=noise_model)
    else:
        if args.noise:
            # Error probabilities
            prob_1 = 0.0005  # 1-qubit gate
            prob_2 = 0.005   # 2-qubit gate
            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(prob_1, 1)
            error_2 = noise.depolarizing_error(prob_2, 2)
            # Add errors to noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            print(noise_model)
            dev = qml.device('qiskit.aer', wires=args.n_qubits, noise_model=noise_model)
        else:
            dev = qml.device("default.qubit", wires=args.n_qubits)

    '''init model'''
    model = CircuitSearchModel(dev, NAS_search_space,Rs_space,CNOTs_space, args.n_qubits, args.n_layers, args.n_experts)

    # a whole circuit
    # cost = qml.VQEcost(lambda params, wires: model(params, wires), hamiltonian, dev)

    step_size = 0.001
    opt = qml.QNGOptimizer(step_size,lam=0.001)

    """
    test for train share-param
    """

    loss_list = []
    resoure_train_share_param = []
    tag = True
    if tag:
        '''train'''
        for epoch in range(args.epochs):
            '''
            # shuffle data
            indices = np.random.permutation(data_train.shape[0])
            data_train = data_train[indices]
            label_train = label_train[indices]
            '''
            subnet = np.random.randint(0, len(NAS_search_space), (args.n_layers,)).tolist()

            @qml.qnode(dev, interface='autograd')
            def circuit_search(params, wires=wires, n_qubits=args.n_qubits, n_layers=args.n_layers, arch=subnet):
                '''
                quantum circuit
                '''
                # nas circuit
                # repeatedly apply each layer in the circuit
                state = [0 for _ in range(args.n_qubits)]
                qml.BasisState(np.array(state), wires=wires)
                for j in range(n_layers):
                    qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])
                return qml.expval(hamiltonian)

            def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]]):
                for i in range(n_qubits):
                    Rs[i](params[j, i], wires=i)
                for conn in CNOTs:
                    qml.CNOT(wires=conn)

            # find the expert with minimal loss w.r.t. subnet
            if epoch < args.warmup_epochs:
                expert_idx = np.random.randint(args.n_experts)
            else:
                expert_idx = expert_evaluator(model, subnet, args.n_experts, circuit_search)  # n_experts: num of the share-param
                resoure_train_share_param.append([subnet,args.n_experts])


            # print('epoch,{}, subnet: {}, expert_idx: {}'.format(epoch,subnet, expert_idx))

            model.params = model.get_params(subnet, expert_idx)

            loss = circuit_search(model.params)
            loss_list.append(loss.item())
            model.params = opt.step(circuit_search,model.params)

            model.set_params(model.params)

            if (epoch+1) %100==0:
                junjian.save_pkl(model.params_space, f'{save_path}/params_space_TFIM{epoch+1}.pkl')
                # exit()


        resoure_search_circuit = []
        '''random'''
        print('Start search.')
        result = {}
        loss_list = []
        subnet_list = []
        if args.searcher == 'random':
            for i in range(args.n_search):
                subnet = np.random.randint(0, len(NAS_search_space), (args.n_layers,)).tolist()

                @qml.qnode(dev, interface='autograd')
                def circuit_search(params, wires=wires, n_qubits=args.n_qubits, n_layers=args.n_layers, arch=subnet):
                    '''
                    quantum circuit
                    '''
                    # nas circuit
                    # repeatedly apply each layer in the circuit
                    state = [0 for _ in range(args.n_qubits)]
                    qml.BasisState(np.array(state), wires=wires)
                    for j in range(n_layers):
                        qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])
                    return qml.expval(hamiltonian)

                def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]]):
                    for i in range(n_qubits):
                        Rs[i](params[j, i], wires=i)
                    for conn in CNOTs:
                        qml.CNOT(wires=conn)

                expert_idx = expert_evaluator(model, subnet, args.n_experts, circuit_search)

                resoure_search_circuit.append([subnet,args.n_experts])

                model.params = model.get_params(subnet, expert_idx)

                energy = circuit_search(model.params)
                # score = -np.abs(energy - (exact_value))
                # result['-'.join([str(x) for x in subnet])] = score
                print('{}/{}: subnet: {}, energy: {}'.format(i+1, args.n_search, subnet, energy))
                loss_list.append(energy.item())
                subnet_list.append(subnet)

        junjian.save_pkl(loss_list,f'{save_path}/loss_list.pkl')
        junjian.save_pkl(subnet_list,f'{save_path}/subnet_list.pkl')


    resoure_retrain = []


    loss_list = junjian.load_pkl(f'{save_path}/loss_list.pkl')
    subnet_list = junjian.load_pkl(f'{save_path}/subnet_list.pkl')

    # trian circuit accroding to ranking
    index = np1.argsort(loss_list)

    # subnet_train = [subnet_list[i] for i in index]
    subnet_train = [subnet_list[index[0]]]

    best_energy = []
    for i_init in range(args.mutil_param_init):
        pass

        for subnet in subnet_train:
            @qml.qnode(dev, interface='autograd')
            def circuit_search(params, wires=wires, n_qubits=args.n_qubits, n_layers=args.n_layers, arch=subnet):
                '''
                quantum circuit
                '''
                # nas circuit
                # repeatedly apply each layer in the circuit
                state = [0 for _ in range(args.n_qubits)]
                qml.BasisState(np.array(state), wires=wires)
                for j in range(n_layers):
                    qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])
                return qml.expval(hamiltonian)


            def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]]):
                for i in range(n_qubits):
                    Rs[i](params[j, i], wires=i)
                for conn in CNOTs:
                    qml.CNOT(wires=conn)


            params = np.random.uniform(0, np.pi * 2, (args.n_layers, n_qubits))
            step_size = 0.01
            # exact_value = -1.136189454088
            opt = qml.QNGOptimizer(step_size,lam=0.001)

            t = 0
            for epoch in range(2000):
                params = opt.step(circuit_search, params)
                energy = circuit_search(params)
                # print(epoch,energy)
                if np.abs(energy.item()-exact_value)<=0.001 or np.abs(t-energy)<0.001:
                    break
                t = energy

            best_energy.append(energy)
        resoure_retrain.append([subnet,epoch])


    return subnet, np.min(best_energy),com_quantum_resource





if __name__ == '__main__':
    # once()
    # exit()

    ####
    # 实际上这里做的是5qubit的海森堡模型，名字没改
    ####

    result = [[],[], []]
    seed_list = [0]
    for seed in seed_list:
        subnet, energy,com_quantum_resource = main(seed)
        print(f"seed{seed}, energy:{energy}")

