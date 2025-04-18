"""
jit版本DQAS，qubit连接性有限制, 任意qubit数, 任意问题
"""

import numpy as np
import tensorcircuit as tc
import tensorflow as tf
from sujun import list_arc_to_structure
import task_VQE
import matplotlib.pyplot as plt


class DQAS():
    def __init__(self, vqef, n, p, ch):
        self.K = tc.set_backend("tensorflow")
        self.ctype, self.rtype = tc.set_dtype("complex128")
        self.vqef = vqef

        # 量子比特数、门数、操作池大小
        """task related config"""
        self.n, self.p, self.ch = n, p, ch

        self.ops_repr = ["h0", "h1", "h2", "h3", "h4", "h5",
                    "rx0", "rx1", "rx2", "rx3", "rx4", "rx5",
                    "ry0", "ry1", "ry2", "ry3", "ry4", "ry5",
                    "rz0", "rz1", "rz2", "rz3", "rz4", "rz5",
                    "xx01", "xx12", "xx23", "xx34", "xx45", "xx50",
                    "yy01", "yy12", "yy23", "yy34", "yy45", "yy50",
                    "zz01", "zz12", "zz23", "zz34", "zz45", "zz50"
                    ]
        # self.edge_list = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        # self.edge_list = [[0, 1], [1, 2], [2, 3]]
        self.edge_list=[[i,i+1] for i in range(self.n-1)] + [[self.n-1,0]]

        self.vag1 = self.K.jit(self.K.vvag(vqef, argnums=0, vectorized_argnums=(1, 2)))

        self.nmf_gradient_vmap = self.K.vmap(self.nmf_gradient, vectorized_argnums=1)

    def sampling_from_structure(self, structures):
        """
        in: stp that gate_wise in SO
        """
        prob = self.K.softmax(self.K.real(structures), axis=-1)
        prob = prob.numpy()
        sampled_cir = np.array([np.random.choice(self.ch, p=prob[i]) for i in range(self.p)])
        one_hot_cir = self.K.onehot(sampled_cir, num=self.ch)
        cir = []
        for g in sampled_cir:
            cir.append([int(g/self.n), g%self.n])
        return cir, one_hot_cir, sampled_cir


    def sampling_from_structure_FA(self, structures):
        """
        in stp that gate_wise in FA
        """
        prob = self.K.softmax(self.K.real(structures), axis=-1)
        prob = prob.numpy()
        sampled_cir = np.array([np.random.choice(self.ch, p=prob[i]) for i in range(self.p)])
        one_hot_cir = self.K.onehot(sampled_cir, num=self.ch)
        cir = []
        for g in sampled_cir:
            cir.append([int(g/self.n), g%self.n])
        return cir, one_hot_cir, sampled_cir


    def best_from_structure(self, structures):
        return self.K.argmax(structures, axis=-1)


    def nmf_gradient(self, structures, oh):
        """
        根据朴素平均场概率模型计算蒙特卡洛梯度
        """
        choice = self.K.argmax(oh, axis=-1)
        prob = self.K.softmax(self.K.real(structures), axis=-1)
        indices = self.K.transpose(self.K.stack([self.K.cast(tf.range(self.p), "int64"), choice]))
        prob = tf.gather_nd(prob, indices)
        prob = self.K.reshape(prob, [-1, 1])
        prob = self.K.tile(prob, [1, structures.shape[1]])

        return tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=self.ctype),
            indices,
            tf.ones([self.p], dtype=self.ctype),
        )


    def process(self,epochs,batch,nnp_lr,stp_lr):

        verbose = False

        # nnp_lr=0.06 stp=0.12
        lr = tf.keras.optimizers.schedules.ExponentialDecay(nnp_lr, 100, 0.5)
        network_opt = tf.keras.optimizers.Adam(lr)

        structure_opt = tf.keras.optimizers.Adam(stp_lr)

        pi = 2 * np.pi
        nnp = np.random.normal(0, 0.02, (self.p, self.ch))
        nnp = tf.Variable(
            initial_value=tf.convert_to_tensor(nnp, dtype=getattr(tf, tc.rdtypestr))
        )

        stp = np.random.normal(0, 0.02, (self.p, self.ch))
        stp = tf.Variable(
            initial_value=tf.convert_to_tensor(stp, dtype=getattr(tf, tc.rdtypestr))
        )

        avcost1 = 0
        averaging_loss_list = []
        searching_loss_list = []
        searching_circuit = []
        for epoch in range(epochs):  # 更新结构参数的迭代
            avcost2 = avcost1
            costl = []
            batched_stuctures = [self.sampling_from_structure(stp) for _ in range(batch)]
            batched_stuctures_for_nnp = self.K.cast([self.list_arc_to_structure_layerwise(batched_stuctures[_][0]) for _ in range(batch)], "int64")
            batched_stuctures_for_stp = self.K.cast(np.stack([batched_stuctures[_][1] for _ in range(batch)]), self.rtype)
            guide = self.K.cast(np.stack([batched_stuctures[_][2] for _ in range(batch)]), "int64")

            infd, gnnp = self.vag1(nnp, batched_stuctures_for_nnp, guide)
            gs = self.nmf_gradient_vmap(stp, batched_stuctures_for_stp)  # \nabla lnp
            gstp = [self.K.cast((infd[i] - avcost2), self.ctype) * gs[i] for i in range(infd.shape[0])]
            gstp = self.K.real(self.K.sum(gstp, axis=0) / infd.shape[0])
            avcost1 = self.K.sum(infd) / infd.shape[0]

            network_opt.apply_gradients([(gnnp, nnp)])
            structure_opt.apply_gradients([(gstp, stp)])

            averaging_loss_list.append(np.average(infd.numpy()))

            if epoch%20==0:
                numpy_stp = stp.numpy()
                s_now = np.argmax(numpy_stp, axis=-1)
                one_hot_cir_now = self.K.onehot(s_now, num=self.ch)
                cir_now = []
                for g in s_now:
                    cir_now.append([int(g / self.n), g % self.n])

                cir_now_struct = self.list_arc_to_structure_layerwise(cir_now)
                cir_now_struct = self.K.cast(cir_now_struct, "int64")
                e_now = self.vqef(nnp, cir_now_struct, s_now)
                searching_circuit.append([cir_now,s_now,cir_now_struct])
                searching_loss_list.append(e_now.numpy())

        return averaging_loss_list, searching_loss_list, searching_circuit


    def process_FA(self,epochs,batch,nnp_lr,stp_lr):
        """
        return a stp that present the better topology, which can shrunk the search for SO
        """
        verbose = False

        # nnp_lr=0.06 stp=0.12
        lr = tf.keras.optimizers.schedules.ExponentialDecay(nnp_lr, 100, 0.5)
        network_opt = tf.keras.optimizers.Adam(lr)

        structure_opt = tf.keras.optimizers.Adam(stp_lr)

        pi = 2 * np.pi
        nnp = np.random.normal(0, 0.02, (self.p, self.ch))
        nnp = tf.Variable(
            initial_value=tf.convert_to_tensor(nnp, dtype=getattr(tf, tc.rdtypestr))
        )

        stp = np.random.normal(0, 0.02, (self.p, self.ch))
        stp = tf.Variable(
            initial_value=tf.convert_to_tensor(stp, dtype=getattr(tf, tc.rdtypestr))
        )

        avcost1 = 0
        averaging_loss_list = []
        searching_loss_list = []
        stp_list = []

        for epoch in range(epochs):  # 更新结构参数的迭代

            avcost2 = avcost1
            batched_stuctures = [self.sampling_from_structure_FA(stp) for _ in range(batch)]
            batched_stuctures_for_nnp = self.K.cast([self.list_arc_to_structure(batched_stuctures[_][0]) for _ in range(batch)], "int64")
            batched_stuctures_for_stp = self.K.cast(np.stack([batched_stuctures[_][1] for _ in range(batch)]), self.rtype)
            guide = self.K.cast(np.stack([batched_stuctures[_][2] for _ in range(batch)]), "int64")

            infd, gnnp = self.vag1(nnp, batched_stuctures_for_nnp, guide)
            gs = self.nmf_gradient_vmap(stp, batched_stuctures_for_stp)  # \nabla lnp
            gstp = [self.K.cast((infd[i] - avcost2), self.ctype) * gs[i] for i in range(infd.shape[0])]
            gstp = self.K.real(self.K.sum(gstp, axis=0) / infd.shape[0])
            avcost1 = self.K.sum(infd) / infd.shape[0]
            network_opt.apply_gradients([(gnnp, nnp)])
            structure_opt.apply_gradients([(gstp, stp)])

            averaging_loss_list.append(np.average(infd.numpy()))

            if epoch%20==0:

                numpy_stp = stp.numpy()
                s_now = np.argmax(numpy_stp, axis=-1)
                one_hot_cir_now = self.K.onehot(s_now, num=self.ch)
                cir_now = []
                for g in s_now:
                    cir_now.append([int(g / self.n), g % self.n])
                cir_now = self.list_arc_to_structure(cir_now)
                cir_now = self.K.cast(cir_now, "int64")
                e_now = self.vqef(nnp, cir_now, s_now)

                searching_loss_list.append(e_now)
                stp_list.append(stp.numpy())
        return averaging_loss_list, searching_loss_list, stp.numpy(), stp_list

    def process_SO(self,epochs,batch,nnp_lr,stp_lr,topology,FA_init=0.015):

        verbose = False

        # nnp_lr=0.06 stp=0.12
        lr = tf.keras.optimizers.schedules.ExponentialDecay(nnp_lr, 100, 0.5)
        network_opt = tf.keras.optimizers.Adam(lr)

        structure_opt = tf.keras.optimizers.Adam(stp_lr)

        pi = 2 * np.pi
        nnp = np.random.normal(0, 0.02, (self.p, self.ch))
        nnp = tf.Variable(
            initial_value=tf.convert_to_tensor(nnp, dtype=getattr(tf, tc.rdtypestr))
        )

        # stp = np.random.normal(0, 0.02, (self.p, self.ch))
        # stp = np.zeros((self.p, self.ch))

        stp = np.full((self.p, self.ch), 1/self.ch)

        for i in range(len(topology)):
            for j in range(len(topology[i])):
                if topology[i][j] != 0:
                    # stp[i][j] = FA_init
                    # stp[i][j] = topology[i][j]
                    stp[i][j] = stp[i][j]*2

        stp = tf.Variable(
            initial_value=tf.convert_to_tensor(stp, dtype=getattr(tf, tc.rdtypestr))
        )
        prob = np.array(self.K.softmax(self.K.real(stp), axis=-1))

        avcost1 = 0
        averaging_loss_list = []
        searching_loss_list = []
        searching_archi = []

        for epoch in range(epochs):  # 更新结构参数的迭代

            avcost2 = avcost1

            batched_stuctures = [self.sampling_from_structure(stp) for _ in range(batch)]
            batched_stuctures_for_nnp = self.K.cast([self.list_arc_to_structure(batched_stuctures[_][0]) for _ in range(batch)], "int64")
            batched_stuctures_for_stp = self.K.cast(np.stack([batched_stuctures[_][1] for _ in range(batch)]), self.rtype)
            guide = self.K.cast(np.stack([batched_stuctures[_][2] for _ in range(batch)]), "int64")

            infd, gnnp = self.vag1(nnp, batched_stuctures_for_nnp, guide)
            gs = self.nmf_gradient_vmap(stp, batched_stuctures_for_stp)  # \nabla lnp
            gstp = [self.K.cast((infd[i] - avcost2), self.ctype) * gs[i] for i in range(infd.shape[0])]
            gstp = self.K.real(self.K.sum(gstp, axis=0) / infd.shape[0])
            avcost1 = self.K.sum(infd) / infd.shape[0]

            network_opt.apply_gradients([(gnnp, nnp)])
            structure_opt.apply_gradients([(gstp, stp)])

            averaging_loss_list.append(np.average(infd.numpy()))
            if epoch%20==0:
                numpy_stp = stp.numpy()
                s_now = np.argmax(numpy_stp, axis=-1)
                one_hot_cir_now = self.K.onehot(s_now, num=self.ch)
                cir_now = []
                for g in s_now:
                    cir_now.append([int(g / self.n), g % self.n])

                cir_struct = self.list_arc_to_structure(cir_now)
                cir_struct = self.K.cast(cir_struct, "int64")
                e_now = self.vqef(nnp, cir_struct, s_now)
                searching_archi.append([cir_now,cir_struct,s_now])
                searching_loss_list.append(e_now.numpy())

        return averaging_loss_list, searching_loss_list, searching_archi

