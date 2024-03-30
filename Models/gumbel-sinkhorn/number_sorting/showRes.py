import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
from utils import makeGMM
from eval import reconstruct

region = 1000
Device = torch.device('cuda')

def show_train_hist(loss, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(loss))
    y = loss

    plt.plot(x, y, label='TFM Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_result(testData, transformer, show = False, save = False, path = 'result.png'):
    result = np.zeros((64,64,64), np.float32)

    x,y,z = 0,0,0
    print('Generating Data:')
    for y_, minmax in tqdm(testData):
        y_ = y_[0]
        minmax = minmax[0]

        xmin, xmax = minmax[0], minmax[1]
        scale = (xmax-xmin)/region
        gmm = makeGMM(np.array(y_))
        samples = torch.Tensor(gmm.sample(64)[0])
        y_label = torch.clamp(( samples - xmin )//scale, 0, region-1).view(1, 64)
        # print(y_label.shape)

        predict = transformer(y_label.to(torch.int64).to(Device), torch.zeros(1, 64).to(torch.int64).to(Device))
        predict = reconstruct(xmin, scale, predict, gmm)

        block = predict.cpu().data.view(4,4,4).numpy()

        block_size = 4
        if z>=64:
            z%=64
            y+=block_size
        if y>=64:
            y%=64
            x+=block_size
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    result[i+x][j+y][k+z] = block[i][j][k]
        z += block_size

    result.tofile(path)