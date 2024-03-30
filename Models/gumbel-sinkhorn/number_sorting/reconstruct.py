import os, time
import torch

import numpy as np
from tqdm import tqdm
from GMM_utils import makeGMM, getTest

import matplotlib.pyplot as plt

import os, sys
from model import FCModel
sys.path.append(os.pardir)
from utils import gumbel_sinkhorn_ops
import position_encodings

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

def recon(num_workers, model, path):
    test_data = getTest(num_workers)
    result = np.zeros((12,12,12), np.float32)
    resample_usingGroundDistribution = False
    resample_usingModel = True

    x,y,z = 0,0,0
    print('Generating Data:')
    for x_,  y_, minmax, xyz in tqdm(test_data):
        xmin, xmax = minmax[0][0], minmax[0][1]
        
        if resample_usingGroundDistribution:
            shuffle_idx = torch.randperm(n_numbers)
            samples = x_.view(-1)[shuffle_idx].view(1, 64)
        else:
            gmm = makeGMM(np.array(y_[0]))
            samples = torch.Tensor(gmm.sample(64)[0]).view(1, 64)
        

        if resample_usingModel:
            X = (samples-xmin)/(xmax-xmin)

            ### sin cos position embedding
            # p_enc_model = position_encodings.PositionalEncoding3D(n_numbers)
            # p_enc_sum = position_encodings.Summer(p_enc_model)
            # X = p_enc_sum(X, batch_size, xyz)
            # X = (X+1)/3

            ### learnable position embedding
            # l_pos_emb = position_encodings.learnable_embedding(n_numbers, n_numbers//3+1)
            # pos_emb = l_pos_emb(torch.LongTensor(xyz.type(torch.int64))).view(1, (n_numbers//3+1) *3)[:, :n_numbers]
            # pos_X = X + pos_emb
            # pos_X = pos_X.to(device)

            X = X.to(device)
            xyz = torch.LongTensor(xyz.type(torch.int64)).to(device)

            log_alpha = model(X[:,None], xyz)
            assingment_matrix = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

            est_permutation = assingment_matrix.max(1)[1].float()
            est_sample = X[:, est_permutation.int()]
            est_sample = est_sample*(xmax-xmin)+xmin

            block = est_sample.cpu().data.view(4,4,4).numpy()
        else:
            block = samples.cpu().data.view(4,4,4).numpy()
            # plt.hist(samples, bins=64)
            # plt.show()

        block_size = 4
        if z>=12:
            z=0
            y+=block_size
        if y>=12:
            y=0
            x+=block_size
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    result[i+x][j+y][k+z] = block[i][j][k]
        z += block_size

        result.tofile(path)

if __name__=='__main__':
    num_workers = 8
    batch_size = 1
    hid_c = 256
    n_numbers = 64

    model = FCModel(hid_c, n_numbers).to(device)
    model.load_state_dict(torch.load('./log/raw6_tau16_ns16_e100_MSE.pth'), strict=False)
    path = './log/bin/block64.bin'

    recon(num_workers, model, path)