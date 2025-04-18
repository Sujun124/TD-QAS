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
        """
        这个函数是专门为qas这种需要频繁切换线路设计的，具体原理不大好讲清楚，看makec里的select_gate函数的实现
        :param param: 单个门的旋转角参数
        :return: 当前门的表示方式的集合，它会被输入make_c的select_gate函数中，最终只会有一个奏效
        这个专门为了这个gate_set下，假如以后要换量子门种类，这个还是作为输入吧，这个作为一个例子放在这里
        """
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
        """
        :param param: 线路参数，由于makec参与jit，必须得是tensor
        :param structure: 线路结构，由于makec参与jit，必须得是tensor
        :return: 返回线路实体，它本质上代表着vqe中的trail state

        根据structure挨个将门加到c上。内层for循环会在每个qubit组合上根据structure作用一个门，structure是one_hot的，除了非0位置对应的
        qubit组合外的组合都只会作用4*4的identity，等于什么都没做，只有非0的那个组合上会作用那个非零数字在gate_list中对应的门。这样每一个内层
        for循环实际上等价于只作用一个门。这样确实增大了线路门数量，提升了计算开销，但由于不再需要if else判断语句，且tensor in tensor out,
        我们可以jit,这将显著缩减优化线路所需时间。
        """
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

