## ContextPrior_Paddle

### 1、简介

ContextPrior Architecture

> 论文原文：[2004.01547.pdf ](https://arxiv.org/pdf/2004.01547.pdf)

![网络结构](https://blog.puqing.work/p/context-prior-for-scene-segmentation%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/CPNet_hu6ad48c13c72a068fc4507e0e9bb0faee_73339_1024x0_resize_q75_box.jpg)

具体网络说明可以移步到本人博客，曾写有论文笔记[ContextPrior论文阅读笔记](https://blog.puqing.work/p/context-prior-for-scene-segmentation%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/)

本项目利用百度的paddlepaddle框架对CVPR2020论文context prior for scene
segmentation的复现。项目依赖于paddleseg工具，因此可以使用paddleseg中提供的训练和评估API进行训练与评估。

### 复现精度

| Model                | mIOU  |
| -------------------- | ----- |
| CPNet50(原论文mmseg) | 44.46 |
| CPNet50              | 45.78  |

### 数据集

使用的数据集为：[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

- Training set：20.210 images
- Validation set：2.000 images