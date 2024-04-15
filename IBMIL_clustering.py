import enum
import re
from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.datasets import build_HDF5_feat_dataset
from architecture.ibmil import IBMIL

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import yaml
from utils.utils import Struct

torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import time
import numpy as np
import faiss
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata


def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]


class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))


def reduce(conf, feats, k):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T)
                           for i in range(k)])

    os.makedirs(f'datasets_deconf/{conf.dataset}/{conf.seed}', exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes = prototypes.reshape(-1, conf.D_inner)
    print(prototypes.shape)
    print(f'datasets_deconf/{conf.dataset}/train_bag_cls_agnostic_feats_proto_{k}_pretrain_%s_seed_%s.npy'%(conf.pretrain, conf.seed))
    np.save(f'datasets_deconf/{conf.dataset}/train_bag_cls_agnostic_feats_proto_{k}_pretrain_%s_seed_%s.npy'%(conf.pretrain, conf.seed), prototypes)

    del feats


def main():
    parser = argparse.ArgumentParser(description='Clutering for abmil/dsmil/transmil')
    parser.add_argument('--config', dest='config', default='config/camelyon17_medical_ssl_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    parser.add_argument('--c_learn', action='store_true', help='learn confounder or not')



    # parser.add_argument('--dir', type=str,help='directory to save logs')
    # dsmil
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')

    args = parser.parse_args()
    # args = parser.parse_args(['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor'])
    '''
    ['--feats_size','512', '--num_classes','1', '--dataset','Camelyon16_Img_nor']
    ['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor']
    
    
    '''

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    milnet = IBMIL(conf)


    # state_dict_weights = torch.load(args.load_path)
    load_path = os.path.join('./saved_models', 'ds_%s_%s_arch_ibmil'%(conf.dataset, conf.pretrain), str(conf.seed), 'checkpoint-best.pth')
    state_dict_weights = torch.load(load_path)['model']
    msg = milnet.load_state_dict(state_dict_weights)
    print("***********loading init from {}*******************".format(load_path))
    print(msg.missing_keys)
    milnet.to(device)
    milnet.eval()

    train_data, _, _ = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)

    # forward
    feats_list = []
    for i, data in enumerate(train_loader):
        with torch.no_grad():
            bag_feats = data['input'].to(device, dtype=torch.float32)
            bag_prediction, bag_feats, attention = milnet(bag_feats)

            feats_list.append(bag_feats.cpu())
    bag_tensor = torch.cat(feats_list, dim=0)

    # bag_tensor=torch.load(f'datasets/{args.dataset}/abmil/ft_feats.pth')
    bag_tensor_ag = bag_tensor.view(-1, conf.D_inner)
    reduce(conf, bag_tensor_ag, 8)


if __name__ == '__main__':
    main()