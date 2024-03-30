import os, time
import torch
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from GMM_utils import makeGMM, my_data_load, getTest
from scipy.stats import wasserstein_distance

import logging
from model import FCModel
import os, sys
sys.path.append(os.pardir)
from utils import gumbel_sinkhorn_ops
import position_encodings

def train(cfg):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    
    model = FCModel(cfg.hid_c, cfg.n_numbers).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)

    train_loader = my_data_load(cfg.batch_size, cfg.num_workers)

    logger.info("training")
    for epoch in range(cfg.epochs):
        for x_batch, y_batch, minmax, xyz in tqdm(train_loader):
            xmin, xmax = minmax[:, 0], minmax[:, 1]
            xmin, xmax = xmin.view(cfg.batch_size, 1), xmax.view(cfg.batch_size, 1)

            ordered_X = (x_batch-xmin)/(xmax-xmin)
            X = ordered_X[:, torch.randperm(ordered_X.size()[1])]

            ### sin cos position embedding
            # p_enc_model = position_encodings.PositionalEncoding3D(cfg.n_numbers)
            # p_enc_sum = position_encodings.Summer(p_enc_model)
            # X = p_enc_sum(X, cfg.batch_size, xyz)
            # X = (X+1)/3

            X = X.to(device)
            xyz = torch.LongTensor(xyz.type(torch.int64)).to('cuda')
            ordered_X = ordered_X.to(device)

            log_alpha = model(X[:,None], xyz)

            gumbel_sinkhorn_mat = [
                gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha, cfg.tau, cfg.n_sink_iter)
                for _ in range(cfg.n_samples)
            ]

            est_ordered_X = [
                gumbel_sinkhorn_ops.inverse_permutation(X, gs_mat)
                for gs_mat in gumbel_sinkhorn_mat
            ]

            loss = sum([
                torch.nn.functional.mse_loss(X, ordered_X)
                # torch.nn.functional.cross_entropy(X, ordered_X)
                for X in est_ordered_X
            ])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info("%i epoch training loss %f", epoch, loss.item())

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, 
        "raw6_tau" + str(cfg.tau) + "_ns" + str(cfg.n_samples) + "_e" + str(cfg.epochs) + "_MSE.pth"
    ))

def evaluation(cfg):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    model = FCModel(cfg.hid_c, cfg.n_numbers).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.out_dir,
        "raw3_tau" + str(cfg.tau) + "_ns" + str(cfg.n_samples) + "_e" + str(cfg.epochs) + "_MSE.pth"
    )))

    test_data = getTest(cfg.num_workers)
    resample_usingGroundValue = False

    logger.info("evaluation")
    model.eval()

    for x_,  y_, minmax, xyz in tqdm(test_data):
        xmin, xmax = minmax[0][0], minmax[0][1]

        if resample_usingGroundValue:
            shuffle_idx = torch.randperm(cfg.n_numbers)
            samples = samples.view(-1)[shuffle_idx].view(1, 64)
        else:
            gmm = makeGMM(np.array(y_[0]))
            samples = torch.Tensor(gmm.sample(64)[0]).view(1, 64)

        ordered_X = (x_-xmin)/(xmax-xmin)
        X = (samples-xmin)/(xmax-xmin)

        X = X.to(device)
        ordered_X = ordered_X.to(device)

        log_alpha = model(X[:,None])
        assingment_matrix = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

        est_permutation = assingment_matrix.max(1)[1].float()
        est_sample = X[:, est_permutation.int()][0]

        est = est_sample[0].cpu().numpy()
        ground = ordered_X[0].cpu().numpy()
        diff = np.subtract(est, ground)
        square = np.square(diff)
        MSE = square.mean()
        RMSE = np.sqrt(MSE)
        print(" RMSE: ", RMSE)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=16, type=float, help="temperture parameter")
    parser.add_argument("--n_sink_iter", default=20, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=16, type=int, help="number of samples from gumbel-sinkhorn distribution")
    # datase option
    parser.add_argument("--n_numbers", default=64, type=int, help="number of sorted numbers")
    parser.add_argument("--train_seed", default=1, type=int, help="random seed for training data generation")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # optimizer option
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=27, type=int, help="mini-batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    # misc
    parser.add_argument("--hid_c", default=256, type=int, help="number of hidden channels")
    parser.add_argument("--out_dir", default="log", type=str, help="/path/to/output directory")
    eval_only = False

    cfg = parser.parse_args()

    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)

    # logger setup
    logging.basicConfig(
        filename=os.path.join(cfg.out_dir, "console.log"),
    )
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger("NumberSorting")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(cfg)

    if not eval_only:
        train(cfg)
    # evaluation(cfg)