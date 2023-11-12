# Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification

This is the Pytorch implementation of our "Attention-Challenging  Multiple Instance Learning for Whole Slide Image Classification". This code is based on the [**CLAM**](https://github.com/mahmoodlab/CLAM/).


# Dataset Preparetion
We provide the features of bracs datasets for the reimplemention of our results. 
Extracted patch features using ImageNet supervised ResNet18: https://drive.google.com/file/d/1VaaaZF_anH46wQG2NxRAt4ENBQuNoXAq/view?usp=drive_link

Extracted patch features using SSL ViT-S/16: https://drive.google.com/file/d/14nYcr7SahYI0xSNVTXnp7u2ju3FSdrOU/view?usp=drive_link

For your own dataset, you can modify and run [Step1_create_patches_fp.py](Step1_create_patches_fp.py) and [Step2_feature_extract.py](Step2_feature_extract.py). More details about this file can refer [**CLAM**](https://github.com/mahmoodlab/CLAM/).
Note that we recommond extracting features using SSL pretrained method. Our code using the checkpoints provided by [Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html)

# Training
For the ABMIL (baseline), you should set n_token=1 n_masked_patch=0 mask_drop=0
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification.py --seed 4 --wandb_mode online --arch ga --n_token 1 --n_masked_patch 0 --mask_drop 0 --config config/bracs_natural_supervised_config.yml
```
For our ACMIL, you should set n_token=5 n_masked_patch=10 mask_drop=0.6
```shell
CUDA_VISIBLE_DEVICES=2 python Step3_WSI_classification.py --seed 4 --wandb_mode online --arch ga --n_token 5 --n_masked_patch 10 --mask_drop 0.6 --config config/bracs_natural_supervised_config.yml
```




