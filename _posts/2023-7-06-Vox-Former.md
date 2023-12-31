---
layout: post
title: "VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion"
categories: [computer vision]
comments: true
---

In this blog post, we will delve into the research paper titled [VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion (CVPR 2023)](https://arxiv.org/abs/2302.12251) authored by Y. Li, Z. Yu, C. Choy, C. Xiao, Jose M. Alvarez, S. Fidler, C. Feng, A. Anandkumar. We will begin with a concise introduction outlining the objective of the study, followed by a discussion on related works to provide a comprehensive overview of the current research landscape concerning semantic scene completion. Subsequently, the results and ablation study will be presented, leading up to the concluding remarks and future prospects.

<div align="center">
  <img img src="/gif/VF_main.gif" alt="Demo Video" width="650">
  <p style="text-align:center;font-style:italic;">Paper demo​.</p>
</div>

# Introduction
We as humans have a remarkable ability to mentally reconstruct the complete three-dimensional (3D) geometry of objects and scenes. Our visual system can leverage a wide range of depth cues such as depth, reflection, etc. This combined with the ability to recognize patterns enables us to connect 2D images with 3D priors of objects that we have seen before. 

<div align="center">
  <img img src="/images/vox/image14.png" alt="Rubic Cube" width="700">
  <p style="text-align:center;font-style:italic;">Various depth cues illustration​.</p>
</div>

Computer vision models face a challenge in replicating the complex cognitive processes of human vision. Understanding how our brains work and programming it into a machine or deep learning model is a daunting task. One approach is to train deep learning models on every 3D priors, such as shape and texture, to reconstruct the 3D scene. However, this approach is impractical. VoxFormer aims to bring us closer to mimicking this remarkable ability, which is called **semantic scene completion (SCC)**. It involves generating a comprehensive 3D representation of a scene, including occupancy and semantic labels, from a single viewpoint such as RGB, LiDAR, or both.

<div style="text-align: center;">
  <table style="width: 100%;">
    <tr>
      <td style="width: 33%; border: none; ">
        <img src="/gif/image19.gif" alt="Image 1" height="150" style="width: 100%;">
        <p style="font-style:italic;">R., Luis; C.e, Raoul de; V., Anne (2020): LMSCNet. Lightweight Multiscale 3D Semantic Completion.</p>
      </td>
      <td style="width: 33%; border: none;">
        <img src="/gif/image17.gif" alt="Image 2" height="150" style="width: 100%;">
        <p style="font-style:italic;">Li, Jie; H., Kai; W., Peng; L., Yu; Y., Xia (CVPR 2020): Anisotropic Convolutional Networks for 3D Semantic Scene Completion​.</p>
      </td>
      <td style="width: 33%; border: none;">
        <img src="/gif/image18.gif" alt="Image 3" height="150" style="width: 100%;">
        <p style="font-style:italic;">C., Anh-Quan; C., Raoul de (CVPR 2021): MonoScene. Monocular 3D Semantic Scene Completion​.</p>
      </td>
    </tr>
  </table>
</div>

# Related Works
In this section, I'm going to present three papers that I think are the most relevant. This will help you to understand the current research landscape and what the VoxFormer does that is different from the previous works. 

## LMSCNet(LiDAR)
<div align="center">
  <img img src="/images/vox/image20.png" alt="LMSCNet" width="1150">
  <p style="text-align:center;font-style:italic;">R., Luis; C.e, Raoul de; V., Anne (2020): LMSCNet. Lightweight Multiscale 3D Semantic Completion.</p>
</div>

**LMSCNet** is included here as it is used as the backbone for stage one of our VoxFormer paper, which will be discussed later. This lidar-based method treats depth as a feature rather than a dimension. This allows the use of established Conv2D techniques. Additionally, it employs the U-net architecture with multi-scale prediction. The result is a high-performance completion model with fast training and inference speed.

## MonoScene (RGB)
<div align="center">
  <img img src="/images/vox/image21.png" alt="MonoScene" width="1150">
  <p style="text-align:center;font-style:italic;">C., Anh-Quan; C., Raoul de (CVPR 2021): MonoScene. Monocular 3D Semantic Scene Completion​.</p>
</div>

**MonoScene**, our main competitor, differs from VoxFormer as it reconstructs the scene using only a single RGB image without relying on 2.5D or 3D information. They employ a projection module called **FLoSP (Feature Line of Sight projection)** to transform 2D features into a unified 3D feature map. However, this approach suffers from feature ambiguity, where empty voxels may receive irrelevant features from the image.

## SCPNet (LiDAR)
<div align="center">
  <img img src="/images/vox/image22.png" alt="SCPNet" width="850">
  <p style="text-align:center;font-style:italic;">X., Zhaoyang; L., Youquan; L., Xin; Z., Xinge; M., Yuexin; L,, Yikang et al. (CVPR 2023): SCPNet. Semantic Scene Completion on Point Cloud.</p>
</div>

Lastly, **SCPNet** is the current leading LiDAR approach. It introduces two key enhancements. Firstly, the completion network is redesigned with **Multi-Path Blocks (MPBs)**, which aggregates point cloud features at multiple scales. Secondly, a **knowledge distillation objective** is employed, using a multi-frame teacher model and a single-frame student model. This transfers dense, relation-based semantic knowledge from the teacher to the student while preserving fast inference speed. 


<div align="center">
  <img img src="/images/vox/image24.png" alt="SCPNet" height="270">
  <img img src="/images/vox/image23.png" alt="SCPNet" height="270">
  <p style="text-align:center;font-style:italic;">X., Zhaoyang; L., Youquan; L., Xin; Z., Xinge; M., Yuexin; L,, Yikang et al. (CVPR 2023): SCPNet. Semantic Scene Completion on Point Cloud.</p>
</div>


# Methodology
## Overview
<div align="center">
  <img img src="/images/vox/image25.png" alt="Overview" width="1150">
  <p style="text-align:center;font-style:italic;">Y., Zhenyu; L., Yifan; C., Chen; C., Chiyu; X., Chen; A., Jose et al. (CVPR 2021): VoxFormer. Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion.</p>
</div>

This is the overview of the VoxFormer paper. As shown in the above figure, it consists of two stages. **Stage 1: Class-Agnostic Query Proposal** generates 3D query inputs. **Stage 2: Class-Specific Segmentation** generates the final semantic output. To delve deeper into VoxFormer's distinct approach, further breakdown is needed.

## Stage 1: Class-Agnostic Query Proposal
<div align="center">
  <img img src="/images/vox/stage1.png" alt="Overview" width="1150">
  <p style="text-align:center;font-style:italic;">Stage 1: Class-Agnostic Query Proposal.</p>
</div>

Using two RGB images from a stereo camera setup, our approach begins by predicting a depth map with **MobileStereoNet**. This depth map is then projected back to 3D space using camera intrinsic parameters, resulting in a point cloud. Voxelization of the point cloud yields a sparse occupancy voxel grid of size $$H\times W\times Z$$, denoting occupied and empty voxels. This **2.5D representation** serves as input for adapted LMSCNet, a fast and high-performing LiDAR-based method for occupancy prediction. The predicted occupancy grid, with lower resolution $$h\times w\times z$$, enhances robustness against noisy depth estimation. This process embodies the paper's first key concept of "**reconstruction before hallucination**".

## Stage 2: Class-Specific Segmentation
<div align="center">
  <img img src="/images/vox/image30.jpg" alt="Overview" width="1150">
  <p style="text-align:center;font-style:italic;">Stage 2: Class-Specific Segmentation.</p>
</div>

In stage two, multiple images from different time stamps are used and processed by a **ResNet-50** backbone to obtain the feature map $$\boldsymbol{F^{2D}_t}$$, which serves as keys and values for the attention module. Only occupied voxels (blue) from the predicted occupancy grid are used as query input, demonstrating the paper's second key contribution of "**sparsity-in-3D-space**". Now, let's address the topic of Deformable Attention.

### Deformable Attention
Each voxel in the occupancy grid corresponds to a point in space, which can be projected back to the image using camera parameters. This forms the basis of deformable attention, where each query only attends to features at its corresponding location on the feature map. This is mathematically expressed as:
<div align="center">
  <p style="text-align:center;">$$\large DA(\boldsymbol{q},\boldsymbol{p},\boldsymbol{F}) = \sum^{N_s}_{s=1}\boldsymbol{A_s}\boldsymbol{W_s}\boldsymbol{F}(\boldsymbol{p}+\delta\boldsymbol{p_s})$$ </p>
</div>

In the above expression, each query $$\boldsymbol{q}$$ is updated with the feature map $$\boldsymbol{F}$$ at the location of the corresponding point $$\boldsymbol{p}$$. Deformable attention extends beyond the corresponding point, also considering the surrounding area (8 surrounding points) using learnable offsets $$\boldsymbol{\delta p_s}$$. Interpolation is used to calculate values at these locations. The attention weight $$\boldsymbol{A_s}$$ is normalized between [0,1], while $$\boldsymbol{W_s}$$ represents the $$\boldsymbol{V}$$ matrix in the standard attention mechanism. Let's explore how this is applied in the paper.
<div align="center">
  <img img src="/images/vox/illustration of DA.png" alt="Overview" width="950">
  <p style="text-align:center;font-style:italic;">Illustration of the deformable attention.</p>
</div>

### Deformable Cross-Attention
As mentioned before, the extracted image features are used as $$\boldsymbol{K}$$, $$\boldsymbol{V}$$ while the occupied voxels are used as $$\boldsymbol{Q}$$. Again, each query $$\boldsymbol{q_p}$$ from the voxel grid will be updated with the feature map $$\boldsymbol{F^{2D}_t}$$ at the location $$\mathcal{P}(\boldsymbol{p},t)$$. The presence of the time stamp $$t$$ in this equation is crucial because each point in space is visible in specific images only. For example, in the image above, the black car within the orange box is visible solely in frames $$t$$ and $$t+1$$, so it makes sense that those queries from that car should only attend to the images where they appear on. Those images are indexed with $$t$$ and grouped as $$\boldsymbol{\mathcal{V}_t}$$. 

### Deformable Self-Attention
Next, the updated queries are combined with mask tokens $$\boldsymbol{m}$$, which account for empty or occluded voxels. Each query $$\boldsymbol{f}$$, representing either a mask token or an updated query proposal, is then updated with the feature map $$\boldsymbol{F^{3D}}$$ at its location and neighboring points. This aggregates image features from the updated queries to the remaining voxel regions. The resulting 3D feature map is upscaled and passed through a few fully connected layers to generate the final semantic output. The loss function employed is the weighted cross-entropy loss.

<div align="center">
  <img img src="/images/vox/stage2.png" alt="Overview" width="950">
  <p style="text-align:center;font-style:italic;">Stage 2: Class-Specific Segmentation with Deformable Attention.</p>
</div>

# Result and Ablation Study
## Dataset and Evaluation Metrics
The paper is evaluated on the SemanticKITTI(2019) SCC with RGB images taken from KITTI Odometry Benchmark with 22 outdoor driving scenarios. The interested volume is 51.2$$m$$ ahead, 6.4$$m$$ in height, and 25.6$$m$$ left and right side. Voxel grid has size 0.2$$m$$ x 0.2$$m$$ x 0.2$$m$$ with 20 classes (1 unknown class). The evaluation matrices are mIoU and IoU.

## Quantitative Result
Examining the results presented below, it is evident that the VoxFormer paper surpasses the current leading camera-based method, MonoScene, by a considerable margin. Notably, as the distance to the ego vehicle decreases, the performance gap between VoxFormer and state-of-the-art LiDAR-based SSC methods reduces significantly, to the extent that VoxFormer even outperforms certain methods in this group.
<div style="text-align: center; border: none; ">
  <table style="width: 100%; border: none; ">
    <tr>
      <td style="width: 40%; border: none; text-align: center; vertical-align: middle;">
        <img src="/images/vox/image47.png" alt="Image 1" width="650">
        <p style="text-align:center;font-style:italic;">       Quantitative comparison against MonoScene.</p>
      </td>
      <td style="width: 40%; border: none; text-align: center; vertical-align: middle;">
        <img src="/images/vox/image48.png" alt="Image 2" width="650">
        <p style="text-align:center;font-style:italic;">Quantitative comparison against LiDAR-based SSC methods. Top three: red, green, and blue</p>
      </td>
    </tr>
  </table>
</div>

## Qualitative Result
Here are the qualitative results of VoxFormer. Upon closer examination of the animation, it becomes evident that when surrounding cars are moving at high speeds, the point cloud's position in the current frame differs significantly from the next frame. Consequently, incorrect projection of these points onto the future frames leads to some cars appearing distorted.
<div align="center">
  <img img src="/gif/VF2.gif" alt="Demo Video" width="1050">
  <p style="text-align:center;font-style:italic;">VoxFormer in action​.</p>
</div>
<div align="center">
  <img img src="/images/vox/image53.png" alt="Demo Video" width="1050">
</div>
<div align="center">
  <img img src="/images/vox/image54.png" alt="Demo Video" width="1050">
  <p style="text-align:center;font-style:italic;">Qualitative comparison against MonoScene and LMSCNet​.</p>
</div>
Finally, the qualitative comparison shows us the superior performance of VoxFormer. In the first example, only VoxFormer is able to correctly complete the shape of the car on the bottom right, whereas MonoScene can detect the car but failed to create a comprehensive shape and LMSCNet doesn't detect it at all. In the second example, VoxFormer is the only model that correctly segmented the pole while the others failed to do so.

## Ablation Study
Next, we will explore the ablation study conducted by the authors to validate the effectiveness of their proposed method. Let's begin with the impact of accurate depth estimation on the model's performance. The results demonstrate the significant difference between depth estimation obtained through a stereo setup (more accurate) and monocular estimation, reflected in a 6-8% improvement in IoU and 2-3% in mIoU.
<div align="center">
  <img img src="/images/vox/image49.png" alt="Overview" width="650">
  <p style="text-align:center;font-style:italic;">Ablation study for depth estimation.</p>
</div>

Moving on, we examine the trade-off of using multiple temporal inputs. While there is no major improvement in IoU performance since there is no change in the query proposal process, incorporating future frames allows objects that are farther in the current frame to appear closer, resulting in a more detailed feature map for queries to attend to, thus boosting mIoU performance. However, this enhancement comes at the expense of increased computational cost, with VRAM usage rising from 15.21 GB to 19.37 GB.
<div align="center">
  <img img src="/images/vox/image50.png" alt="Overview" width="650">
  <p style="text-align:center;font-style:italic;">Ablation study for temporal input.</p>
</div>

Lastly, the table below highlights the remarkable benefits of the selected query proposal method. It not only outperforms dense query proposals by approximately 10% in IoU but also reduces GPU memory consumption by 4 GB.
<div align="center">
  <img img src="/images/vox/image51.png" alt="Overview" width="650">
  <p style="text-align:center;font-style:italic;">Ablation study for query proposal.</p>
</div>

# Conclusion
## Contribution
The VoxFormer paper demonstrates cutting-edge results in camera-based semantic scene completion, surpassing even certain lidar-based approaches. Its fundamental concepts of "**reconstruction before hallucination**" and "**sparsity-in-3D-space**" have not only proven to enhance performance but also optimize GPU memory usage. The inclusion of deformable attention further amplifies these benefits.

## Limitation
Nevertheless, some limitations still exist. Firstly, the paper's performance in long-range scenarios remains weak, as depth estimation lacks reliability in corresponding locations. However, considering the model's performance solely based on estimated depth, it raises curiosity about the potential heights it could achieve if given an accurate measurement.
The last constraint pertains to the model's unreliability when objects in the surroundings move at high speeds, caused by significant deviations in the point cloud pose between current and future frames. 

# Bibliography
- [[1]](https://arxiv.org/abs/2112.00726)Cao, Anh-Quan; Charette, Raoul de (CVPR 2021): MonoScene. Monocular 3D Semantic Scene Completion.
- [[2]](https://arxiv.org/abs/1908.02893)Dourado, Aloisio; Campos, Teofilo Emidio de; Kim, Hansung; Hilton, Adrian (2019): EdgeNet. Semantic Scene Completion from a Single RGB-D Image.
- [[3]](https://arxiv.org/abs/2302.12251)Li, Yiming; Yu, Zhiding; Choy, Christopher; Xiao, Chaowei; Alvarez, Jose M.; Fidler, Sanja et al. (CVPR 2023): VoxFormer. Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion.
- [[4]](https://arxiv.org/abs/2008.10559)Roldão, Luis; Charette, Raoul de; Verroust-Blondet, Anne (2020): LMSCNet. Lightweight Multiscale 3D Semantic Completion.
- [[5]](https://arxiv.org/abs/2303.06884)Xia, Zhaoyang; Liu, Youquan; Li, Xin; Zhu, Xinge; Ma, Yuexin; Li, Yikang et al. (CVPR 2023): SCPNet. Semantic Scene Completion on Point Cloud.
- [[6]](https://arxiv.org/abs/2010.04159)Zhu, Xizhou; Su, Weijie; Lu, Lewei; Li, Bin; Wang, Xiaogang; Dai, Jifeng (2020): Deformable DETR. Deformable Transformers for End-to-End Object Detection.
- [[7]](https://arxiv.org/abs/2004.02122)Li, Jie; Han, Kai; Wang, Peng; Liu, Yu; Yuan, Xia (CVPR 2020): Anisotropic Convolutional Networks for 3D Semantic Scene Completion.
- [[8]](https://arxiv.org/abs/1904.01416)Behley, Jens; Garbade, Martin; Milioto, Andres; Quenzel, Jan; Behnke, Sven; Stachniss, Cyrill; Gall, Juergen (2019): SemanticKITTI. A Dataset for Semantic Scene Understanding of LiDAR Sequences.
- [[9]](https://arxiv.org/abs/2108.09770)Faranak Shamsafar, Samuel Woerz, Rafia Rahim, Andreas Zell (2021): MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching


