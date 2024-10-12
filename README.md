# Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification (ECCV2024)

This is the Pytorch implementation of our "[Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/pdf/2311.07125.pdf)". This code is based on the [**CLAM**](https://github.com/mahmoodlab/CLAM/).

# News
**[2024.10]** In our experiments, we found that setting ``mask_drop=0`` sometimes yielded better performance than setting ``mask_drop=0.6``. If you find that ACMIL performance is not stable enough, you can try setting mask_drop=0.0.

**[2024.8]** We are excited to introduce Attention Entropy Maximization (AEM), a novel plug-and-play regularization technique designed to address attention concentration in Multiple Instance Learning (MIL) frameworks. AEM offers a simple yet effective solution for mitigating overfitting in whole slide image classification tasks, requiring no additional modules and featuring just one hyperparameter. This innovative approach demonstrates excellent compatibility with various MIL frameworks and techniques. We invite researchers and practitioners to explore our GitHub repository at [https://github.com/dazhangyu123/AEM](https://github.com/dazhangyu123/AEM) and delve into the details in our paper available on arXiv: [https://arxiv.org/abs/2406.15303](https://arxiv.org/abs/2406.15303).

**[2024.8]** We are thrilled to announce the publication of the official repository for "PathGen-1.6M: 1.6 Million Pathology Image-text Pairs Generation through Multi-agent Collaboration". This groundbreaking project introduces a novel approach to generating high-quality pathology image-text pairs using multi-agent collaboration. Researchers and practitioners in pathology and machine learning can now access the PathGen-1.6M dataset, explore the implementation of the multi-agent system, and utilize the pre-trained PathGen-CLIP model. We invite you to visit the repository at [https://github.com/superjamessyx/PathGen-1.6M](https://github.com/superjamessyx/PathGen-1.6M) to dive into this valuable resource and contribute to advancing pathology-specific vision language models. For more details about the project, please refer to our paper on [arXiv](https://arxiv.org/pdf/2407.00203).

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
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 1 --n_masked_patch 0 --mask_drop 0 --pretrain natural_supervised --config config/bracs_config.yml
```
For our ACMIL, you can run [Step3_WSI_classification_ACMIL.py](Step3_WSI_classification_ACMIL.py) and set n_token=5 n_masked_patch=10 mask_drop=0.6
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_ACMIL.py --seed 4 --wandb_mode online --arch ga --n_token 5 --n_masked_patch 10 --mask_drop 0.6 --pretrain natural_supervised --config config/bracs_config.yml
```
For CLAM, DAMIL, and TransMIL, you run [Step3_WSI_classification.py](Step3_WSI_classification.py) 
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification.py --seed 4 --wandb_mode online --arch clam_sb/clam_mb/transmil/dsmil --pretrain natural_supervised --config config/bracs_config.yml
```
For DTFD-MIL, you run [Step3_WSI_classification_DTFD.py](Step3_WSI_classification_DTFD.py) 
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification_DTFD.py --seed 4 --wandb_mode online --pretrain natural_supervised --config config/bracs_config.yml
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


