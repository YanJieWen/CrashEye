<div align="center">
  <img src="assets/CrashEye-logo-removebg-preview.png" width="400"/>
  <div>&nbsp;</div>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 📖 目录
- [简介](#简介)
- [安装](#安装)
- [数据准备](#数据准备)
- [模型库](#模型库)
- [运行说明](#运行说明)
- [案例](#案例)
- [注意事项](#注意事项)
- [致谢](#致谢)
- [许可证](#许可证)

## 简介
[CrashEye](https://github.com/YanJieWen/CrashEye)是第一个面向**列车耐撞性**设计的目标检测+跟踪项目。它可以在无需任何人工干预的情况下实现列车碰撞过程中的棋盘导航点的自主轨迹提取。

主分支代码目前在``Pytorch 2.0`` ``Python 3.8`` ``CUDA 11.8``版本上运行。  

- **集成多个检测/跟踪架构**

CrashEye集成的检测器包括[MMdetection](https://github.com/open-mmlab/mmdetection)/[ultralytics](https://github.com/ultralytics/ultralytics)/强基线[CVMR](https://github.com/YanJieWen/CVMR)。
跟踪器包括[MMTracking](https://github.com/open-mmlab/mmtracking)/强基线[MASORT](https://github.com/YanJieWen/MASORT)。另外，我们的强基线还包括一个自监督的Re-ID模块[LWTGPF](https://github.com/YanJieWen/LWTGPF-2025)。
开发人员可以组合不同的组件和模块来自定义模型。  


- **多场景适应性**

CrashEye在多源碰撞场景中是有效的，包括端部吸能碰撞，整车碰撞，缩比列车碰撞等。  
<div align="center">
<img src="assets/trackres.gif" width="600"/>
</div>

- **速度快**

所有的操作（检测/跟踪）均在强大的GPU上（```RTX4090@24G```）实现。借助既有的数据处理流水线，无需对高速摄影仪（HS2C）采集的高分辨率图像进行处理，而是通过缩放操作来显著降低显存。  
<div align="center">
<img src="assets/data_pipeline.png" />
</div>

- **性能高**
CrashEye包含一个强基线模型，采用[CVMR](https://github.com/YanJieWen/CVMR)来实现小目标检测，[LWTGPF](https://github.com/YanJieWen/LWTGPF-2025)进行实例的深度外观特征提取，[MASORT](https://github.com/YanJieWen/MASORT)对目标进行跨帧的自适应跟踪。

## 安装




