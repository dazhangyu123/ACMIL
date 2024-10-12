
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

from utils.utils import cosine_scheduler, Struct, set_seed, Wandb_Writer, ema_update, save_model
from datasets.datasets import build_HDF5_feat_dataset
from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from utils.utils import get_cam_1d
import torchmetrics
from timm.utils import accuracy
from copy import deepcopy
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--wandb_mode', default='disabled', choices=['offline', 'online', 'disabled'],
                        help='the model of wandb')
    parser.add_argument('--grad_clipping', default=5, type=float)

    parser.add_argument('--model', default='pure', type=str, help='Model name')


    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='attn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0.1, type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=0.5, type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)

    # Siamese framework
    parser.add_argument('--cl_alpha', default=0.1, type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='./saved_models/ds_camelyon_medical_ssl_arch_pure/2/checkpoint-best.pth', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

    # Misc
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')

    parser.add_argument('--pretrain', default='medical_ssl',
                        choices=['natural_supervised', 'medical_ssl', 'path-clip-L-336'],
                        help='settings of Tip-Adapter in yaml format')
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )

    args = parser.parse_args()
    return args

def train_one_epoch(model, model_tea, criterion, data_loader, optimizer,
                    device, epoch, conf, mm_sche):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()
    if model_tea is not None:
        model_tea.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)


        # for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        bag = data['input'].to(device, dtype=torch.float32)
        label = data['label'].to(device)
        batch_size = bag.shape[0]


        if conf.model == 'mhim':
            if model_tea is not None:
                cls_tea, attn = model_tea.forward_teacher(bag, return_attn=True)
            else:
                attn, cls_tea = None, None

            cls_tea = None if conf.cl_alpha == 0. else cls_tea

            train_logits, cls_loss, patch_num, keep_num = model(bag, attn, cls_tea, i=epoch * len(data_loader) + data_it)
        elif conf.model == 'pure':
            train_logits, cls_loss, patch_num, keep_num = model.pure(bag)
            cls_loss = torch.tensor(0.)
        else:
            train_logits = model(bag)
            cls_loss, patch_num, keep_num = torch.tensor(0.), 0., 0.

        logit_loss = criterion(train_logits.view(batch_size, -1), label)
        train_loss = conf.cls_alpha * logit_loss + cls_loss * conf.cl_alpha


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if conf.model == 'mhim':
            if mm_sche is not None:
                mm = mm_sche[epoch * len(data_loader) + data_it]
            else:
                mm = conf.mm
            if model_tea is not None:
                if conf.tea_type == 'same':
                    pass
                else:
                    ema_update(model, model_tea, mm)
        else:
            mm = 0.


        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(logit_loss=logit_loss.item())
        metric_logger.update(cls_loss=cls_loss.item())

        if conf.wandb_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            wandb.log({'cls_loss': cls_loss}, commit=False)
            wandb.log({'logit_loss': logit_loss})

# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, header):

    # Set the network to evaluation mode
    model.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        bag = data['input'].to(device, dtype=torch.float32)
        label = data['label'].to(device)
        batch_size = bag.size(0)

        if conf.model in ('mhim', 'pure'):
            test_logits = model.forward_test(bag)
        elif conf.model == 'dsmil':
            test_logits, _ = model(bag)
        else:
            test_logits = model(bag)

        test_logits = test_logits.view(batch_size, -1)
        pred_softmax = torch.softmax(test_logits, dim=-1)
        loss = criterion(test_logits, label)
        acc1 = accuracy(pred_softmax, label, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=1)


        y_pred.append(pred_softmax)
        y_true.append(label)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


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
    elif conf.pretrain == 'natural_supervsied':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-L-336':
        conf.D_feat = 768
        conf.D_inner = 384


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ADR",
        # track hyperparameters and run metadata
        config={'dataset': conf.dataset,
                'pretrain': conf.pretrain,
                'loss_form': conf.model,
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
    mm_sche = None
    if args.model == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h, 0., epochs=conf.train_epoch, niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio': args.mask_ratio,
            'n_classes': conf.n_class,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
            'feat_dim': conf.D_feat,
            'mlp_dim': conf.D_inner
        }
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm, args.mm_final, epochs=conf.train_epoch, niter_per_ep=len(train_loader),
                                       start_warmup_value=1.)


        model = mhim.MHIM(**model_params).to(device)

    elif args.model == 'pure':
        model = mhim.MHIM(select_mask=False,n_classes=conf.n_class,act=args.act,head=args.n_heads,da_act=args.da_act,
                          baseline=args.baseline,feat_dim=conf.D_feat, mlp_dim=conf.D_inner).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=conf.n_class,dropout=args.dropout,act=args.act,feat_d=conf.D_feat).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(conf).to(device)

    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=conf.n_class,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=conf.n_class,dropout=args.dropout,act=args.act).to(device)

    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=conf.n_class,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)

    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=conf.n_class,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=conf.n_class,dropout=args.dropout,act=args.act).to(device)


    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(args.teacher_init)['model']
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        if not args.no_log:
            print(info)


    # teacher model
    if args.model == 'mhim':
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            if not args.no_log:
                print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(args.teacher_init)['model']
                info = model_tea.load_state_dict(pre_dict, strict=False)
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        if args.tea_type == 'same':
            model_tea = model
    else:
        model_tea = None

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr,  weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(conf.train_epoch):

        train_one_epoch(model, model_tea, criterion, train_loader, optimizer, device, epoch, conf, mm_sche)

        # if args.model == 'mhim':
        #     val_auc, val_acc, val_f1, val_loss = evaluate(model_tea, criterion, val_loader, device, conf, 'Val')
        #     test_auc, test_acc, test_f1, test_loss = evaluate(model_tea, criterion, test_loader, device, conf, 'Test')
        # else:
        val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, device, conf, 'Test')

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
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                       save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))
        print('\n')

    save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
               save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))
    print("Results on best epoch:")
    print(best_state)

    wandb.finish()


if __name__ == '__main__':
    main()
