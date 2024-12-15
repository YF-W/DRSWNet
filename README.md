# Application of Dynamic Graph Convolution Networks Integrating Multi-Domain Features in Multi-Lesion Medical Image Segmentation

Paper Address : https://github.com/YF-W/MDDG-Net

Although convolutional neural networks (CNNs) have made significant strides in medical image segmentation, they still face key challenges. These include difficulty in capturing long-range dependencies, limiting the integration of distant features, and reduced segmentation accuracy. Additionally, CNNs are less robust to noise, leading to decreased precision in noisy medical images. While effective at local feature extraction, CNNs struggle with modeling complex spatial structures, particularly in multi-scale images, where capturing fine spatial dependencies is essential.  

## Paper : MDDG Net(Application of Dynamic Graph Convolution Networks Integrating Multi-Domain Features in Multi-Lesion Medical Image Segmentation)

Authors : Yufei Wang, 

## Architecture Overview

![Overview](D:\科研\论文\papper\XNet\Hatmap\Overview.png)  

The MDDG Net introduces a dual-encoding and dual-decoding architecture that integrates both frequency-domain and spatial-domain feature processing. It employs adaptive frequency-domain filtering (ATF) and hierarchical residual operations (HR) for multi-scale, multi-directional feature extraction, enhancing the model’s ability to capture intricate image details. The network incorporates the DC-CRD Module to capture inter-channel correlations and the WT-MFP Module for dynamic downsampling, improving flexibility and generalization. These innovations enable the model to efficiently process complex medical images and resolve fine structural details.

## Our network Baseline

![Baseline](D:\科研\论文\papper\XNet\Hatmap\Baseline.png)

The Baseline model features an innovative "X"-shaped dual-encoder and dual-decoder architecture with separate frequency and spatial domain branches, enhancing multi-task processing and classification performance. The frequency domain branch uses Adaptive Three-Dimensional Frequency Domain Filtering (ATF) to extract refined frequency and directional features, while the spatial domain branch employs hierarchical residual operations (HR) for multi-level spatial feature extraction. ATF integrates frequency-domain filtering, directional enhancement, and dynamic switching between spatial and frequency domains to improve feature selectivity and extraction across domains.

## Module 1: DC-CRD

![DC-CRD](D:\科研\论文\papper\XNet\Hatmap\DC-CRD.png)

The DC-CRD Module introduces three key innovations: dynamically constructing and adaptively adjusting inter-channel relationships, capturing subtle differences between channels via graph convolution, and gradually fusing channel relations with feature maps for more flexible feature propagation. These enhancements improve the model’s ability to capture complex channel dependencies and strengthen feature representation.

## Module 2: WP-MFP

![WT-MFP](D:\科研\论文\papper\XNet\Hatmap\WT-MFP.png)

The WT-WFP Module improves downsampling by precisely processing multi-band features from wavelet transforms, introducing a learnable median filtering mechanism, and dynamically adjusting the importance of different frequency bands with adjustable weight parameters.

## Datasets: 

1. The CELL dataset:https://gitcode.com/gh_mirrors/20/2018DSB

2. The DRIVE dataset:https://drive.grand-challenge.org/

3. The LIVER dataset:https://www.kaggle.com/datasets/zxcv2022/digital-medical-images-for--download-resource

4. The Synapse dataset:https://github.com/Beckschen/TransUNet

5. The ACDC dataset:https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
