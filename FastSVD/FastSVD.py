import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import cupy as cp

class FastSVD(nn.Module):
    def __init__(
        self,
         X, W, r:int=16,
         batch_size:int=5000,
         lr:float=0.001,
         regu=None,
         lamb=None,
         cross=None
    ):
        '''
        Args:
        :param X: numpy array on CPU, [BL, din]
        :param W: numpy array on CPU, [din, dout]
        :param r: rank limitation
        :param regu: int, number of U's columns to sample for standardization of column close to 1
        :param lamb: float, lambda for regularization
        :param cross: int, number of U's columns to sample for column vector orthogonalization
        '''
        super(FastSVD, self).__init__()
        self.gpu_id = cp.cuda.runtime.getDeviceCount() - 1  # 计算最后一个GPU的编号
        self.r = r
        self.BL = X.shape[0]
        self.din = W.shape[0]
        self.dout = W.shape[1]

        # 构建参数
        self.XW = cp.asnumpy(cp.dot(X, W))  # 相乘并转化为numpy数组，形状[BL, dout]

        # # 先注释掉下面一行
        # Sr, VTr = self.precompute(self.XW, r) 

        # # for test
        # print("--Create FastSVD object--")
        # print("XW:", self.XW)
        # print('\n')

        # 将所有数据类型转化为tensor
        # self.Sr = torch.tensor(Sr, dtype=torch.double, requires_grad=False)  # [r, r]
        # self.VTr = torch.tensor(VTr, dtype=torch.double, requires_grad=False)  # [r, dout]
        # self.U = nn.Parameter(torch.randn(self.BL, r, dtype=torch.double))  # 创建可训练参数

        # 训练参数
        self.regu = regu  # U列向量模一化正则项采样数量
        self.lamb = lamb  # 正则化参数
        self.cross = cross  # U列向量正交化的采样对数
        self.batch_size = batch_size  # 训练的批次
        self.lr = lr
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    # 得到初步的分解结果
    def precompute(self, XW, r):
        '''
        Args:
        :param XW: numpy array, [Bl, dout]
        :param r: rank limitation
        '''
        # print("GPU id:", self.gpu_id)

        # 指定设备并执行计算
        with cp.cuda.Device(self.gpu_id):
            XW = cp.asarray(XW)
            S, V = cp.linalg.eigh(cp.dot(XW.T, XW))  # [dout, dout]矩阵特征值分解

            # 查看nan值的数量
            nan_count = cp.sum(cp.isnan(S))
            if nan_count > 0:
                raise ValueError(f'Nan value. Number: {nan_count}')

            # 降序排列
            indices = cp.argsort(S)[::-1]  # 获取特征值排序的索引
            S = S[indices]  # 排序特征值
            V = V[:, indices]  # 排序特征向量

            # 修正半正定实对称矩阵XW.T @ XW特征值中的负值
            S[S < 0] = 0.0

            '''
            下面计算的过程中如果使用直接计算的方式，将会由于S对角线上出现0值而导致S的逆矩阵计算错误，于是需要进行设计：
            首先我们设S中非0的元素的数量为n，则原始矩阵的乘法能够写为：[BL, n] * [n, n] * [n, dout]，所以说这里需要检查n和r的大小，
            判断r是否大于n，r只能取小于等于n的值。
            '''
            n = cp.sum(S > 0)
            if r >= n:
                print(f"The rank of the matrix S is {n}, r will be reset to {n}")
                r = n

            # 构建原始矩阵
            S = cp.sqrt(cp.diag(S[:r]))  # 创建对角矩阵，形状[r, r]
            VT = (V.T)[:r, :]  # 形状[r, dout]

            return cp.asnumpy(S), cp.asnumpy(VT)

    # 直接计算得到
    def ComputeU(self):
        S, VT = self.precompute(self.XW, self.dout)  # 直接计算出全量奇异值分解
        U = self.XW @ VT.T @ np.linalg.inv(S)

        '''
        XW = U @ S @ VT
        '''
        # print("comput U:", U, U.shape)
        # print("compute S:", S, S.shape)
        # print("compute VT:", VT, VT.shape)
        # print("compute INV S:", np.linalg.inv(S))
        # print('\n')

        # 查看逼近程度
        delta = np.abs(self.XW - U @ S @ VT)
        print("SVD(XW) Max bias:", np.max(delta))
        print("SVD(XW) Average bias:", np.sum(delta) / (delta.shape[0] * delta.shape[1]))
        print('\n')
        del delta

        return U, S, VT

    # 生成一个batch的数据对
    def batchsample(self, batch):
        # 构建样本对
        size = (batch, )
        Urow = torch.randint(0, self.BL, size)
        VTcol = torch.randint(0, self.dout, size)

        # 进行采样
        VTbatch = self.VTr[:, VTcol]  # [r, B], tensor
        Ubatch = self.U[Urow, :]  # [B, r], tensor
        target = torch.tensor(self.XW[Urow, VTcol], dtype=torch.double, requires_grad=False) # [B, B], tensor
        return Ubatch, VTbatch, target

    # 损失函数使用MSE，能够添加模一化正则项
    def Loss(self, fit, target):
        loss = torch.mean((fit-target)**2)
        if self.regu is None:
            return loss
        else:
            size = (self.regu, )
            Ucol = torch.randint(0, self.r, size)
            Uregu = self.U[:, Ucol]  # [BL, regu], tensor
            Uregu = torch.sum(Uregu**2, axis=0)  # 沿列求和[regu]
            one  = torch.ones(self.regu, dtype=torch.double, requires_grad=False)
            regular = torch.mean((Uregu - one)**2)
            return loss + self.lamb * regular

    # 训练
    def train(self, epoch):
        avg_loss = 0
        for i in range(epoch):
            Ub, VTb, tar = self.batchsample(self.batch_size)
            fit = Ub @ self.Sr @ VTb
            loss = self.Loss(fit, tar)

            # 训练
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss = i / (i+1) * (avg_loss) + loss.item() / (i+1)

            # 打印损失
            if (i + 1) % 2000 == 0:
                print(f"Epoch [{i + 1}/{epoch}], AVGLoss: {avg_loss:.4f}")