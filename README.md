# Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification (ECCV2024)

This is the Pytorch implementation of our "[Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/pdf/2311.07125.pdf)". This code is based on the [**CLAM**](https://github.com/mahmoodlab/CLAM/).

# News
**[2024.7]** We provide a stronger feature encoder for WSI classification, pre-trained by Vision-Language alignment on PathGen-1.6M. For more details, please refer to our  paper [PathGen-1.6M: 1.6 Million Pathology Image-text Pairs
Generation through Multi-agent Collaboration](https://arxiv.org/pdf/2407.00203)

**[2024.7]** We restructured the entire project to improve readability and modified the wandb setup to better manage the experiment logging.


# Dataset Preparation
We provide a part of the extracted features to reimplement our results. 


## Camelyon16 Dataset (20× magnification)

| Model | Download Link |
|-------|---------------|
| ImageNet supervised ResNet18 | [Download](https://pan.quark.cn/s/dd77e6a476a0) |
| SSL ViT-S/16 | [Download](https://pan.quark.cn/s/6ea54bfa0e72) |
| PathGen-CLIP ViT-L (336 × 336 pixels) | [Download](https://pan.quark.cn/s/62fe3dc65291) |

## Bracs Dataset

### 10× magnification

| Model | Download Link |
|-------|---------------|
| ImageNet supervised ResNet18 | [Download](https://pan.quark.cn/s/7cf21bbe46a7) |
| SSL ViT-S/16 | [Download](https://pan.quark.cn/s/f2f9c93cd5e1) |

### 20× magnification

| Model | Download Link |
|-------|---------------|
| ImageNet supervised ResNet18 | [Download](https://pan.quark.cn/s/cbe4e1d0e68c) |
| SSL ViT-S/16 | [Download](https://pan.quark.cn/s/3c8c1ffce517) |
| PathGen-CLIP ViT-L (336 × 336 pixels) | [Download](https://pan.quark.cn/s/62fe3dc65291) |

For your own dataset, you can modify and run [Step1_create_patches_fp.py](Step1_create_patches_fp.py) and [Step2_feature_extract.py](Step2_feature_extract.py). More details about this file can refer [**CLAM**](https://github.com/mahmoodlab/CLAM/).
Note that we recommend extracting features using SSL pretrained method. Our code using the checkpoints provided by [Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html)

# Training
For the ABMIL (baseline), you can run [Step3_WSI_classification_ACMIL.py](Step3_WSI_classification_ACMIL.py) and set n_token=1 n_masked_patch=0 mask_drop=0
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 1 --n_masked_patch 0 --mask_drop 0 --config config/bracs_natural_supervised_config.yml
```
For our ACMIL, you can run [Step3_WSI_classification_ACMIL.py](Step3_WSI_classification_ACMIL.py) and set n_token=5 n_masked_patch=10 mask_drop=0.6
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 5 --n_masked_patch 10 --mask_drop 0.6 --config config/bracs_natural_supervised_config.yml
```
For CLAM, DAMIL, and TransMIL, you run [Step3_WSI_classification.py](Step3_WSI_classification.py) 
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification.py --seed 4 --wandb_mode online --arch clam_sb/clam_mb/transmil/dsmil --config config/bracs_natural_supervised_config.yml
```
For DTFD-MIL, you run [Step3_WSI_classification_DTFD.py](Step3_WSI_classification_DTFD.py) 
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_DTFD.py --seed 4 --wandb_mode online --config config/bracs_natural_supervised_config.yml
```

## BibTeX
If you find our work useful for your project, please consider citing the following paper.


```
@misc{zhang2023attentionchallenging,
      title={Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification}, 
      author={Yunlong Zhang and Honglin Li and Yuxuan Sun and Sunyi Zheng and Chenglu Zhu and Lin Yang},
      year={2023},
      eprint={2311.07125},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


