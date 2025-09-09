<div align="center">
  <img src="assets/CrashEye-logo-removebg-preview.png" width="400"/>
  <div>&nbsp;</div>
</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 📖 目录
- [简介](#简介)
- [新闻](#新闻)
- [安装](#安装)
- [数据准备](#数据准备)
- [模型库](#模型库)
- [运行说明](#运行说明)
- [案例](#案例)
- [注意事项](#注意事项)
- [致谢](#致谢)
- [许可证](#许可证)

## 新闻
- **⭐2025-09-08**  
  - 初始版本 `CrashEye` 发布，支持列车碰撞序列图像自主感知。  
  - 提供 `Crash2024` `Crash-ReID` `Crash-Seq` 数据集接口。
  - 提供各个模型预训练权重，包括检测以及Re-ID。 

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

CrashEye包含一个强基线模型，采用[CVMR](https://github.com/YanJieWen/CVMR)来实现小目标检测，[LWTGPF](https://github.com/YanJieWen/LWTGPF-2025)进行实例的深度外观特征提取，[MASORT](https://github.com/YanJieWen/MASORT)对目标进行跨帧的自适应跟踪。各基线的性能均在项目库中进行报告。

## 安装  

### 1. 安装CrashEye
```shell
git clone  https://github.com/YanJieWen/CrashEye.git
python setup.py develop
```
### 2. 安装[openmmlab](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html)基线库  
```shell
pip install -U openmim
mim install mmenegine
```
- **安装mmcv**
mmdet与mmcv的版本需要对齐，[参考](https://github.com/open-mmlab/mmtracking/blob/master/docs/zh_cn/install.md)。对于[mmcv](https://github.com/open-mmlab/mmcv)可采用轮子下载，参考列表在[此处](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)
```shell
mim install xxx.whl
```
- **安装mmdetection**
```shell
cd modeling
cd mmdetection
git checkout tags/v2.28.0
python setup.py develop
pip install -v -e .
cd ..
```
- **安装mmtracking**
```shell
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
git checkout tags/v0.14.0
python setup.py  develop
```
### 3. 安装Ultralytics  












