"""
jit版本DQAS，qubit连接性有限制, 任意qubit数, 任意问题
"""

import numpy as np
import tensorcircuit as tc
import tensorflow as tf
import pickle


class VQE():
    def __init__(self,gate_list_in=None):
        self.K = tc.set_backend("tensorflow")
        self.ctype, self.rtype = tc.set_dtype("complex128")
        self.edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]


        # 量子比特数、门数、操作池大小
        """task related config"""
        self.n, self.p, self.ch = 6, 36, 42

        if not gate_list_in:
            self.gate_list = self.gate_list_sample
        else:
            self.gate_list = gate_list_in

        lattice = tc.templates.graphs.Line1D(self.n, pbc=True)
        self.h = tc.quantum.heisenberg_hamiltonian(lattice, hzz=1, hxx=0, hyy=0, hx=1, hy=0, hz=0, sparse=False)  # TFIM
        # h = tc.quantum.heisenberg_hamiltonian(lattice, hzz=1, hxx=1, hyy=0, hx=1, hy=0, hz=0, sparse=False)  # hei
        # self.h = tc.quantum.heisenberg_hamiltonian(lattice, hzz=1, hxx=0, hyy=0, hx=1, hy=0, hz=0, sparse=False)  # LIH

        with open("ham_LIH2.pkl", "rb") as f:
            hamiltonian_dense = pickle.load(f)


        self.h = tf.squeeze(hamiltonian_dense)


    def gate_list_sample(self, param):
        l = [
            tc.gates.Gate(self.K.eye(4)),
            tc.gates.Gate(self.K.kron(tc.gates._h_matrix.astype('complex128'), self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.rx_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.ry_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.rz_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._yy_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._zz_matrix)]
        return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]

    def makec(self, param, structure, g):
        c = tc.Circuit(self.n)
        for i in range(structure.shape[0]):
            for j in range(structure.shape[1]):
                c.select_gate(structure[i][j], self.gate_list(param[i][j]),
                              self.edge_list[j][0], self.edge_list[j][1])
        return c


    def vqef(self, param, structure, g):
        """
        算一条线路在当前参数下对应trail state能得到的能量值，一般这个函数会被jit之后执行
        :param param: 线路参数
        :param structure: 线路结构
        :return: 当前参数下能量值
        """
        # print(f'compiling...')
        c = self.makec(param, structure, g)
        # s = c.state()
        # e = K.sum(K.abs(target - s))
        e = tc.templates.measurements.operator_expectation(c, self.h)

        return e


    def train_process(self, structure, g,epochs=600,lr=1e-2):

        self.vag1 = self.K.jit(self.K.vvag(self.vqef, argnums=0, vectorized_argnums=(1, 2)))

        loss_list = []
        for archi in range(len(structure)):
            network_opt = tc.backend.optimizer(tf.keras.optimizers.Adam(lr))
            nnp = self.K.implicit_randn(stddev=0.02, shape=[self.p, self.ch], dtype=self.rtype)
            nnp = tf.Variable(nnp)
            for epoch in range(epochs):
                infd, gnnp = self.vag1(nnp, structure[archi], g)
                nnp = network_opt.update(gnnp, nnp)

                if epoch % 60 == 0 or epoch == epochs - 1:
                    print(epoch, "loss: ", np.average(infd.numpy()))
            loss_list.append(infd.numpy())

        return loss_list

