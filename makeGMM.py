import numpy as np
import matplotlib.pyplot as plt
import os
from  sklearn.neighbors import KernelDensity as KDE
from sklearn.mixture import GaussianMixture as GMM
from tqdm import tqdm

datatype = {
    'Isabelle'  : (500, 500, 100,),
    'Nyx'       : ( 64,  64,  64,),#(256, 256, 256,),
    'RedSea'    : ()
}


blocksize = 4
reconstruct = True

dname = 'Nyx'
Path = './' + dname + '/makeGMM_input'
savedir= './' + dname + '/block28_36'

target = 'density1.bin'

allFolderList = [f for f in os.listdir(Path)]
allFileList = []
for folder in allFolderList:
    p = os.path.join(Path, folder)
    FileList = [os.path.join(p, f) for f in os.listdir(p)]
    allFileList += FileList

print('Start to Transform from raw_Nyx to GMM(mean, covariance, weight):')
for file in tqdm(allFileList):
    filepath = os.path.join(file,target)
    _, Pram, Time = file.split('\\')
    Time = Time[3:]

    outx = open(os.path.join(savedir, 'x_'+ Pram + '_' + Time + '.bin'), 'w')
    outy = open(os.path.join(savedir, 'y_'+ Pram + '_' + Time + '.bin'), 'w')

    data = np.fromfile(filepath, dtype='float32')
    data = data.reshape(datatype[dname])
    # print(f'data shpae: {data.shape}')

    for i in range(0, datatype[dname][0], blocksize):
        for j in range(0, datatype[dname][1], blocksize):
            for k in range(0, datatype[dname][2], blocksize):
                if 28<=i<=36 and 28<=j<=36 and 28<=k<=36:
                    xmax = min(i+blocksize, datatype[dname][0])
                    ymax = min(j+blocksize, datatype[dname][1])
                    zmax = min(k+blocksize, datatype[dname][2])
                    block_ = data[i:xmax, j:ymax, k:zmax]
                    block = block_.reshape(-1,1)
                    n_components = 5
                    best_gmm = GMM(n_components=n_components, max_iter=100)
                    best_gmm.fit(block)

                    xyz = [i,j,k]
                    min_max = [min(block.flatten()), max(block.flatten())]

                    np.savetxt(outy, xyz, newline=',')
                    outy.write('\n')
                    np.savetxt(outy, min_max, newline=',')
                    outy.write('\n')
                    np.savetxt(outy, best_gmm.means_.flatten(), newline=',')
                    outy.write('\n')
                    np.savetxt(outy, best_gmm.covariances_.flatten(), newline=',')
                    outy.write('\n')
                    np.savetxt(outy, best_gmm.weights_.flatten(), newline=',')
                    outy.write('\n')

###################### 或許更改輸出方式 ######################
                    np.savetxt(outx, block, newline=',')
                    outx.write('\n')
                    # block_.tofile(os.path.join(savedir, 'x_'+ Pram + '_' + Time + '.bin'))
    outx.close()
    outy.close()