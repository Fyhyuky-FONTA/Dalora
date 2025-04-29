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

        # # for SGD method
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
            Compute SVD(XW) with rank constraint of r.
            Args:
            :param XW: numpy array, [Bl, dout]
            :param r: rank limitation
        '''
        # choose specific GPU to perform calculations
        with cp.cuda.Device(self.gpu_id):
            XW = cp.asarray(XW)
            S, V = cp.linalg.eigh(cp.dot(XW.T, XW))  # [dout, dout] matrix eigenvalue decomposition

            # Check the number of Nan value
            nan_count = cp.sum(cp.isnan(S))
            if nan_count > 0:
                raise ValueError(f'Nan value. Number: {nan_count}')

            # sort
            indices = cp.argsort(S)[::-1]  # get the indices of sorted eigenvalues
            S = S[indices]  # sort eigenvalues
            V = V[:, indices]  # sort eigenvectors

            S[S < 0] = 0.0  # correct the negative eigenvalues

            '''
                In the following calculation process, using a direct computation method may lead to errors in 
                calculating the inverse of S due to the presence of 0 values on the diagonal of S. 
                Therefore, a design is necessary: First, let n be the number of non-zero elements in S. 
                The multiplication of the original matrices can be expressed as: [BL, n] * [n, n] * [n, dout]. 
                Thus, it is important to check the sizes of n and r to determine whether r is greater than n. 
                The value of r can only be less than or equal to n.
            '''
            n = cp.sum(S > 0)
            if r >= n:
                print(f"The rank of the matrix S is {n}, r will be reset to {n}")
                r = n

            # get Sigma matrix and V.T matrix
            S = cp.sqrt(cp.diag(S[:r]))  # create diagonal matrix in shape of [r, r]
            VT = (V.T)[:r, :]  # [r, dout]

            return cp.asnumpy(S), cp.asnumpy(VT)

    def ComputeU(self):
        '''Directly compute SVD(XW) with fast svd decomposition.'''
        S, VT = self.precompute(self.XW, self.dout)
        U = self.XW @ VT.T @ np.linalg.inv(S)

        return U, S, VT

    def batchsample(self, batch):
        '''Generate a batch of data pairs.'''
        # building sample element
        size = (batch, )
        Urow = torch.randint(0, self.BL, size)
        VTcol = torch.randint(0, self.dout, size)

        # sample
        VTbatch = self.VTr[:, VTcol]  # [r, B], tensor
        Ubatch = self.U[Urow, :]  # [B, r], tensor
        target = torch.tensor(self.XW[Urow, VTcol], dtype=torch.double, requires_grad=False)  # [B, B], tensor
        return Ubatch, VTbatch, target

    def Loss(self, fit, target):
        '''The loss function uses MSE and can add a L1 regularization term.'''
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

    def train(self, epoch):
        '''Compute SVD(XW) with SGD method.'''
        avg_loss = 0
        for i in range(epoch):
            Ub, VTb, tar = self.batchsample(self.batch_size)
            fit = Ub @ self.Sr @ VTb
            loss = self.Loss(fit, tar)

            # Training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss = i / (i+1) * (avg_loss) + loss.item() / (i+1)

            # print loss
            if (i + 1) % 2000 == 0:
                print(f"Epoch [{i + 1}/{epoch}], AVGLoss: {avg_loss:.4f}")