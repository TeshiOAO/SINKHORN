import os
import torch

import torch.utils.data as Data
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


def getTest(num_workers):
    Path = '..\\..\\..\\..\\Nyx\\GMMDATA\\block_28-36\\TestData'
    allfile = os.listdir(Path)
    x_data, y_data, minmax, xyz = [], [], [], []
    for f in allfile:
        # print(f)
        if f.startswith('y'):
            filetxt = open(os.path.join(Path,f), "r")
            while filetxt:
                l = filetxt.readline()
                if l == "": break
                _xyz = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                _minmax = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                means = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                covs = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                weights = [float(v) for v in l.split(',')[:-1]]
                y_data.append([means, covs, weights])
                minmax.append(_minmax)
                xyz.append(_xyz)
        elif f.startswith('x'):
            filetxt = open(os.path.join(Path,f), "r")
            for l in filetxt.readlines():
                block = np.array([float(v) for v in l.split(',')[:-1]])
                # block = block.reshape((4,4,4,))
                x_data.append(block)

    x_data, y_data, minmax, xyz = torch.Tensor(np.array(x_data)), torch.Tensor(np.array(y_data)), torch.Tensor(np.array(minmax)), torch.Tensor(np.array(xyz))
    torch_dataset = Data.TensorDataset(x_data, y_data, minmax, xyz)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=1,
        num_workers=num_workers
    )
    return loader

def makeGMM(y_data):
    means, covs, weight = [np.array(d) for d in y_data]
    gmm = GMM(n_components=5)
    gmm.means_ = means.reshape(5, 1)
    gmm.weights_ = weight
    gmm.covariances_ = covs.reshape(5, 1, 1)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, 'full')

    return gmm

def my_data_load(batch_size, num_workers):
    Path = '..\\..\\..\\..\\Nyx\\GMMDATA\\block_28-36\\out'
    allfile = os.listdir(Path)
    x_data, y_data, minmax, xyz = [], [], [], []
    for f in allfile:
        # print(f)
        if f.startswith('y'):
            filetxt = open(os.path.join(Path,f), "r")
            while filetxt:
                l = filetxt.readline()
                if l == "": break
                _xyz = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                _minmax = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                means = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                covs = [float(v) for v in l.split(',')[:-1]]
                l = filetxt.readline()
                weights = [float(v) for v in l.split(',')[:-1]]
                y_data.append([means, covs, weights])
                minmax.append(_minmax)
                xyz.append(_xyz)
        elif f.startswith('x'):
            filetxt = open(os.path.join(Path,f), "r")
            for l in filetxt.readlines():
                block = np.array([float(v) for v in l.split(',')[:-1]])
                # block = block.reshape((4,4,4,))
                x_data.append(block)

    x_data, y_data, minmax, xyz = torch.Tensor(np.array(x_data)), torch.Tensor(np.array(y_data)), torch.Tensor(np.array(minmax)), torch.Tensor(np.array(xyz))
    torch_dataset = Data.TensorDataset(x_data, y_data, minmax, xyz)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader

def getZ(mean, cov, n):
    std = np.sqrt(cov.item())
    return mean + std*n


def RescaleDataRange(x, label):
        std_num = 3
        batch = x.shape[0]

        y = label.view(batch, 3, 5)
        for n in range(batch):
            z = torch.sort(y[n][0])[1]
            min_idx, max_idx = z[0], z[-1]
            # print(y[n][0][min_idx], y[n][0][max_idx])
            
            min_val, max_val = getZ(y[n][0][min_idx], y[n][1][min_idx], -std_num), getZ(y[n][0][max_idx], y[n][1][max_idx], std_num)
            val_dif = max_val-min_val

            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        x[n, 0, i, j, k] = min_val + (x[n, 0, i, j, k]+1)/2 * val_dif
        return x

def getPermu(permu_idx, batch_size, n):
    Permu_mtrxs = []
    for k in range(batch_size):
        P = torch.zeros(n, n).type(torch.float)
        for i, index in enumerate(permu_idx[k].long()):
            P[i, index] = 1
        Permu_mtrxs.append(P)

    return Permu_mtrxs