
# !/usr/bin/env python
import sys
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed, Wandb_Writer
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import TransformWrapper, AttnMIL
from architecture.transMIL import TransMIL
from engine import train_one_epoch, evaluate
from architecture.dsmil import MILNet, FCLayer, BClassifier
from architecture.bmil import probabilistic_MIL_Bayes_spvis
from architecture.clam import CLAM_SB, CLAM_MB
from modules import mean_max

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_medical_ssl_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--n_shot", type=int, default=-1, help="number of wsi images"
    )
    parser.add_argument(
        "--w_loss", type=float, default=1.0, help="number of query token"
    )
    parser.add_argument(
        "--arch", type=str, default='transmil', choices=['transmil', 'clam_sb', 'clam_mb', 'attnmil',
                                                 'selfattn', 'dsmil', 'bmil_spvis', 'meanmil', 'maxmil'], help="number of query token"
    )
    parser.add_argument(
        "--n_token", type=int, default=1, help="number of query token"
    )
    parser.add_argument(
        "--n_masked_patch", type=int, default=0, help="whether use adversarial mask"
    )
    args = parser.parse_args()
    return args

def main():
    # Load config file
    args = get_arguments()

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)


    group_name = 'ds_%s_%s_arch_%s_%sepochs' % (conf.dataset, conf.pretrain, conf.arch, conf.train_epoch)
    log_writer = Wandb_Writer(group_name=group_name, mode=args.wandb_mode, name=args.seed)
    conf.ckpt_dir = log_writer.wandb.dir[:-5] + 'saved_models'
    if conf.wandb_mode == 'disabled':
        conf.ckpt_dir = os.path.join(conf.ckpt_dir, group_name, str(args.seed))
    os.makedirs(conf.ckpt_dir, exist_ok=True)
    print("Used config:");
    pprint(vars(conf));

    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    if conf.arch == 'transmil':
        net = TransMIL(conf)
    elif conf.arch == 'selfattn':
        net = TransformWrapper(conf)
    elif conf.arch == 'clam_sb':
        net = CLAM_SB(conf).to(device)
    elif conf.arch == 'clam_mb':
        net = CLAM_MB(conf).to(device)
    elif conf.arch == 'dsmil':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        net = MILNet(i_classifier, b_classifier)
    elif conf.arch == 'bmil_spvis':
        net = probabilistic_MIL_Bayes_spvis(conf)
        net.relocate()
    elif conf.arch == 'attnmil':
        net = AttnMIL(conf)
    elif conf.arch == 'meanmil':
        net = mean_max.MeanMIL(conf).to(device)
    elif conf.arch == 'maxmil':
        net = mean_max.MaxMIL(conf).to(device)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=conf.lr, weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(net, criterion, train_loader, optimizer, device, epoch, conf, log_writer)


        val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, device, conf, 'Test')

        if log_writer is not None:
            log_writer.log('perf/val_acc1', val_acc, commit=False)
            log_writer.log('perf/val_auc', val_auc, commit=False)
            log_writer.log('perf/val_f1', val_f1, commit=False)
            log_writer.log('perf/val_loss', val_loss, commit=False)
            log_writer.log('perf/test_acc1', test_acc, commit=False)
            log_writer.log('perf/test_auc', test_auc, commit=False)
            log_writer.log('perf/test_f1', test_f1, commit=False)
            log_writer.log('perf/test_loss', test_loss, commit=False)

        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            # log_writer.summary('best_acc', val_acc)
            save_model(
                conf=conf, model=net, optimizer=optimizer, epoch=epoch, is_best=True)
        print('\n')

    save_model(
        conf=conf, model=net, optimizer=optimizer, epoch=epoch, is_last=True)
    print("Results on best epoch:")
    print(best_state)


if __name__ == '__main__':
    main()
