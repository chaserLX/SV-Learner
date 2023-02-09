# SV-Learner: Support Vector-drived Contrastive Learning for Robust Learning with Noisy labels

<img src="https://github.com/chaserLX/SV-Learner/blob/main/figures/framework.png"  width="800px" />

This is the official PyTorch implementation of IJCAI2023 paper (SV-Learner: Support Vector-drived Contrastive Learning for Robust Learning with Noisy labels).

**Authors:** Xin Liang, Yanli Ji, Wangmeng Zuo.

**Affliations:** UESTC, HIT.

## Abstract
Noisy-label data inevitably gives rise to confusion in various perception applications. In this paper, we propose a robust-to-noise framework SV-Learner to solve the problem of recognition with noisy labels. In particular, we first design a Dynamic Noisy Sample Selection (DNSS) solution for learning more robust classification boundaries, which dynamically determines the filter rates of classifiers for reliable noisy sample selection based on curriculum learning. Inspired by support vector machines (SVM), we propose a Support Vector driven Contrastive Learning (SVCL) approach that mines support vectors near classification boundaries as negative samples to drive contrastive learning. These support vectors expand the margin between different classes for contrastive learning, therefore better promoting the robust detection of noise samples. Finally, a Dynamic Semi-Supervised Classification (DSSC) module is presented to realize noisy-label recognition. In comparison with the state-of-the-art approaches, the proposed SV-Learner achieves the best performance in multiple datasets, including the CIFAR-10, CIFAR-100, Clothing1M, and Webvision datasets. Extensive experiments demonstrate the effectiveness of our proposed method. 

## Preparation
- numpy
- opencv-python
- Pillow
- torch
- torchnet
- sklearn

Our code is written by Python, based on Pytorch (Version ≥ 1.6).

## Datasets

For CIFAR datasets, one can directly run the shell codes.

For Clothing1M and Webvision, you need to download them from their corresponsing website.

## Usage
Example runs on CIFAR10 dataset with 20% symmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 --lambda_u=0
```

Example runs on CIFAR100 dataset with 90% symmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 --lambda_u=150
```

Example runs on CIFAR10 dataset with 40% asymmetric noise:
```
  python Train_cifar_sv-learner.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'asym' --r 0.4 --lambda_u=25
```

