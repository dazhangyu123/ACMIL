import numpy as np
from pprint import pprint
import yaml
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from utils.utils import save_model, Struct, set_seed, Wandb_Writer
import h5py
import time
from architecture.transformer import ACMIL_GA
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.transMIL import TransMIL
import torch
from wsi_core.WholeSlideImage import WholeSlideImage
import sys

def get_arguments():
    parser = argparse.ArgumentParser('Heatmap visualization', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument('--data_slide_dir', type=str, default='/mnt/Xsky/zyl/dataset/CAMELYON16/training')
    parser.add_argument('--slide_ext', type=str, default='.tif')
    parser.add_argument(
        "--arch", type=str, default='ga', choices=['transmil', 'clam_sb', 'clam_mb',
                                                 'ga'], help="architecture"
    )


    # It is advisable to perform a preliminary check before executing the code.
    parser.add_argument(
        "--seed", type=int, default=2, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--device', default=3, type=int, help="CUDA device")
    parser.add_argument(
        "--n_masked_patch", type=int, default=0, help="whether use adversarial mask"
    )
    parser.add_argument(
        "--n_token", type=int, default=1, help="number of query token"
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0.2, help="number of query token"
    )
    parser.add_argument("--zoom_factor", type=float, default=1.0,
                        help="determine the magnitude of zoom during visualizing the heatmap, range between 0 and 1")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    # Load config file
    args = get_arguments()
    device = 'cuda:{}'.format(args.device)


    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        print("Used config:"); pprint(c);
        conf = Struct(**c)

    group_name = 'ds_%s_%s_arch_attnmil_ntoken_%s_nmp_%s' % (conf.dataset, conf.pretrain, conf.n_token, conf.n_masked_patch)
    ckpt_pth = os.path.join('./saved_models', group_name, str(conf.seed), 'checkpoint-last.pth')
    vis_dir = os.path.join('./vis', group_name)
    os.makedirs(vis_dir, exist_ok=True)


    # define network
    if conf.arch == 'transmil':
        net = TransMIL(conf)
    elif conf.arch == 'ga':
        net = ACMIL_GA(conf)
    elif conf.arch == 'clam_sb':
        net = CLAM_SB(conf, dropout=True)
    elif conf.arch == 'clam_mb':
        net = CLAM_MB(conf, dropout=True)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)


    checkpoint = torch.load(ckpt_pth)
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    net.eval()

    h5_data = h5py.File(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5'%conf.pretrain), 'r')
    slide_names = list(h5_data.keys())
    train_val_names = []
    test_names = []
    for name in slide_names:
        if 'test' in name:
            test_names.append(name)
        else:
            train_val_names.append(name)

    for slide_id in test_names:
        slide_data = h5_data[slide_id]
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        wsi_object = WholeSlideImage(slide_file_path)

        try:
            wsi_object.initXML(os.path.splitext(slide_file_path)[0] + '.xml')
        except:
            print('no xml annos found')
            pass

        feat = torch.from_numpy(slide_data['feat'][:]).unsqueeze(dim=0).to(device, dtype=torch.float32)
        coords = slide_data['coords'][:]

        _, _, attn_scores = net(feat, use_attention_mask=False)

        # visualize the heatmap of ACMIL
        output_path = os.path.join(vis_dir, slide_id + '.png')
        probs = torch.softmax(attn_scores, dim=-1)[0].mean(0).cpu().numpy()
        probs = probs * probs.size * conf.zoom_factor
        heatmap = wsi_object.visHeatmap(scores= probs * 100, coords=coords, patch_size=(512, 512), segment=False,
                                        cmap='jet')
        heatmap.save(output_path)

        # # visualizing the heatmap of each branch in MBA
        # for i in range(conf.n_token):
        #     probs = torch.softmax(attn_scores, dim=-1)[0,i].cpu().numpy()
        #     heatmap = wsi_object.visHeatmap(scores=probs * probs.size * 100, coords=coords, patch_size=(512, 512),
        #                                     segment=False, cmap='jet')
        #     output_path = os.path.join(vis_dir, slide_id + '_%s.png'%i)
        #     heatmap.save(output_path)









if __name__ == '__main__':
    main()


