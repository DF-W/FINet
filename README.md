# Frequency-aware Interaction Network for Ultrasound Image Segmentation

> **Authors:**
> [Dongfang Wang](),
> [Tao Zhou](https://taozh2017.github.io/),
> [Yizhe Zhang](https://yizhezhang.com/), 
> [Shangbing Gao](), and 
> [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=en), 

## 1. Preface

- This repository provides code for "_**Frequency-aware Interaction Network for Ultrasound Image Segmentation (FINet)**_". 
([paper](https://ieeexplore.ieee.org/document/10787068))

## 2. Overview

### 2.1. Introduction

Accurate segmentation of medical ultrasound images is crucial for guiding treatment decisions and assessing intervention effectiveness. The challenge of segmenting lesions in ultrasound images arises from factors such as low contrast, high speckle noise, artifacts, and blurred boundaries. Furthermore, this complexity varies significantly among lesions in different cases. While methods based on Convolutional Neural Networks (CNNs) and Transformers have shown promising results in this field, each approach possesses distinct advantages and limitations. To address these challenges, we propose a novel Frequency-aware Interaction Network (FINet). At the core of our FINet lies the proposed Multi-scale Frequency-aware Self-attention (MFS) module, which effectively captures multi-scale feature information within the self-attention layer. This enables our network to model both local and global features, capitalizing on the strengths of both CNNs and Transformers. Additionally, a frequency-aware network is introduced to learn the interactions between spatial locations in the frequency domain to enhance detailed feature representation such as edges. Furthermore, we present a collaborative interactive decoder network, in which a Selective Feature Interaction (SFI) module is proposed to facilitate the semantic and boundary feature interaction, resulting in more precise segmentation outcomes. Experimental results on four medical ultrasound image datasets show the superiority of our FINet over other state-of-the-art segmentation methods. More importantly, our model achieves an excellent trade-off between performance and computational efficiency.

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/framework.png"/> <br />
    <em> 
    Figure 1: Overview of the proposed CFANet.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/qualitative_results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>
