# Dynamic Graph Convolutional Networks with Multi-Domain Feature Fusion for Multiclass Lesion Segmentation in Medical Imaging

Paper Address : https://github.com/YF-W/MDDG-Net  

Although convolutional neural networks (CNNs) have made significant strides in medical image segmentation, they still face key challenges. These include difficulty in capturing long-range dependencies, limiting the integration of distant features, and reduced segmentation accuracy. Additionally, CNNs are less robust to noise, leading to decreased precision in noisy medical images. While effective at local feature extraction, CNNs struggle with modeling complex spatial structures, particularly in multi-scale images, where capturing fine spatial dependencies is essential.    

## Paper : MDDG Net(Dynamic Graph Convolutional Networks with Multi-Domain Feature Fusion for Multiclass Lesion Segmentation in Medical Imaging)  

Authors : Yufei Wang, 

## Architecture Overview

![Overview](https://github.com/user-attachments/assets/46f64ce0-7756-4b19-ab31-055f375a6434)
 

The MDDG Net introduces a dual-encoding and dual-decoding architecture that integrates both frequency-domain and spatial-domain feature processing. It employs adaptive frequency-domain filtering (ATF) and hierarchical residual operations (HR) for multi-scale, multi-directional feature extraction, enhancing the model’s ability to capture intricate image details. The network incorporates the DC-CRD Module to capture inter-channel correlations and the WT-MFP Module for dynamic downsampling, improving flexibility and generalization. These innovations enable the model to efficiently process complex medical images and resolve fine structural details.  

## Our network Baseline

![Baseline](https://github.com/user-attachments/assets/6c845821-e13d-4e7a-80b9-6d95a427cd21)
  

The Baseline model features an innovative "X"-shaped dual-encoder and dual-decoder architecture with separate frequency and spatial domain branches, enhancing multi-task processing and classification performance. The frequency domain branch uses Adaptive Three-Dimensional Frequency Domain Filtering (ATF) to extract refined frequency and directional features, while the spatial domain branch employs hierarchical residual operations (HR) for multi-level spatial feature extraction. ATF integrates frequency-domain filtering, directional enhancement, and dynamic switching between spatial and frequency domains to improve feature selectivity and extraction across domains.    

## Module 1: DC-CRD

![DC-CRD](https://github.com/user-attachments/assets/3f9f9c25-e7ed-4636-b2cf-868350af0454)
  

The DC-CRD Module introduces three key innovations: dynamically constructing and adaptively adjusting inter-channel relationships, capturing subtle differences between channels via graph convolution, and gradually fusing channel relations with feature maps for more flexible feature propagation. These enhancements improve the model’s ability to capture complex channel dependencies and strengthen feature representation.  

## Module 2: WP-MFP

![WT-MFP](https://github.com/user-attachments/assets/91849337-3aff-423a-bea9-35f690c74b52)
  

The WT-WFP Module improves downsampling by precisely processing multi-band features from wavelet transforms, introducing a learnable median filtering mechanism, and dynamically adjusting the importance of different frequency bands with adjustable weight parameters.

## Datasets: 

1. The CELL dataset:https://gitcode.com/gh_mirrors/20/2018DSB

2. The DRIVE dataset:https://drive.grand-challenge.org/

3. The LIVER dataset:https://www.kaggle.com/datasets/zxcv2022/digital-medical-images-for--download-resource

4. The Synapse dataset:https://github.com/Beckschen/TransUNet

5. The ACDC dataset:https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
