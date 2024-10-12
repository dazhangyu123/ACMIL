
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

from utils.utils import save_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import MHA, ABMIL, ACMIL
from architecture.transMIL import TransMIL
from engine import train_one_epoch, evaluate
from architecture.dsmil import MILNet, FCLayer, BClassifier
from architecture.bmil import probabilistic_MIL_Bayes_spvis
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.ilra import ILRA
from modules import mean_max
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/bracs_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument(
        "--seed", type=int, default=5, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument(
        "--w_loss", type=float, default=1.0, help="number of query token"
    )
    parser.add_argument(
        "--arch", type=str, default='acmil', choices=['transmil', 'clam_sb', 'clam_mb', 'abmil', 'ilra',
                                                 'mha', 'dsmil', 'bmil_spvis', 'meanmil', 'maxmil', 'acmil'], help="number of query token"
    )
    parser.add_argument('--pretrain', default='GigaPath',
                        choices=['natural_supervsied', 'medical_ssl', 'plip', 'path-clip-B-AAAI'
                                                                              'path-clip-B', 'path-clip-L-336',
                                 'openai-clip-B', 'openai-clip-L-336', 'quilt-net', 'biomedclip', 'path-clip-L-768',
                                 'UNI', 'GigaPath'],
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
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

    if conf.pretrain == 'medical_ssl':
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain == 'natural_supervised':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-B' or conf.pretrain == 'openai-clip-B' or conf.pretrain == 'plip'\
            or conf.pretrain == 'quilt-net'  or conf.pretrain == 'path-clip-B-AAAI'  or conf.pretrain == 'biomedclip':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-L-336' or conf.pretrain == 'openai-clip-L-336':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'UNI':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat = 1536
        conf.D_inner = 768


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="wsi_classification",
        # track hyperparameters and run metadata
        config={'dataset': conf.dataset,
                'pretrain': conf.pretrain,
                'arch': conf.arch,
                'seed': conf.seed,},
        mode=conf.wandb_mode
    )
    run_dir = wandb.run.dir  # Get the wandb run directory
    print('Wandb run dir: %s'%run_dir)
    ckpt_dir = os.path.join(os.path.dirname(os.path.normpath(run_dir)), 'saved_models')
    os.makedirs(ckpt_dir, exist_ok=True)  # Create the 'ckpt' directory if it doesn't exist

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
    elif conf.arch == 'mha':
        net = MHA(conf)
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
    elif conf.arch == 'abmil':
        net = ABMIL(conf)
    elif conf.arch == 'acmil':
        net = ACMIL(conf)
    elif conf.arch == 'meanmil':
        net = mean_max.MeanMIL(conf).to(device)
    elif conf.arch == 'maxmil':
        net = mean_max.MaxMIL(conf).to(device)
    elif conf.arch == 'ilra':
        net = ILRA(feat_dim=conf.D_feat, n_classes=conf.n_class, ln=True)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # define optimizer, lr not important at this point
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=conf.lr, weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(net, criterion, train_loader, optimizer, device, epoch, conf)


        val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, device, conf, 'Test')

        if conf.wandb_mode != 'disabled':
            wandb.log({'test/test_acc1': test_acc}, commit=False)
            wandb.log({'test/test_auc': test_auc}, commit=False)
            wandb.log({'test/test_f1': test_f1}, commit=False)
            wandb.log({'test/test_loss': test_loss}, commit=False)
            wandb.log({'val/val_acc1': val_acc}, commit=False)
            wandb.log({'val/val_auc': val_auc}, commit=False)
            wandb.log({'val/val_f1': val_f1}, commit=False)
            wandb.log({'val/val_loss': val_loss}, commit=False)

        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            # log_writer.summary('best_acc', val_acc)
            save_model(conf=conf, model=net, optimizer=optimizer, epoch=epoch,
                       save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
        print('\n')

    save_model(conf=conf, model=net, optimizer=optimizer, epoch=epoch,
               save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))
    print("Results on best epoch:")
    print(best_state)


    wandb.finish()


if __name__ == '__main__':
    main()
