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
mmdet与mmcv的版本需要对齐，[参考此处](https://github.com/open-mmlab/mmtracking/blob/master/docs/zh_cn/install.md)。对于[mmcv](https://github.com/open-mmlab/mmcv)可采用轮子下载，参考列表在[此处](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)

[![mmcv](https://img.shields.io/badge/mmcv-1.7.2-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing)

```shell
mim install mmcv_full-1.7.2-cp38-cp38-manylinux1_x86_64.whl
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
cd ..
```
### 3. 安装Ultralytics  

- **安装CVMR**
```shell
git clone https://github.com/YanJieWen/CVMR.git
cd CVMR
```
- **安装[Casual Conv1D](https://github.com/Dao-AILab/causal-conv1d/releases)和[SSM](https://github.com/state-spaces/mamba/releases)算子**

[![Casual Conv1D](https://img.shields.io/badge/CNN-cuda-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing)
[![SSM](https://img.shields.io/badge/Mamba-cuda-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing)

```shell
pip install causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install -v -e .
```
- **修改源码**

将modeling/CVMR/ultralytics/engine/validator.py 第`131`行强制半精度注释 （在某些请款下训练可能导致Nan）

### 4. 安装MASORT及其外部库

```shell
cd modeling/MASORT
python setup.py develop
```

### 5. 下载external库，可参考[MASORT](https://github.com/YanJieWen/MASORT)

```shell
cd external
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
pip install -v -e .
cd ..

git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py develop
cd ..

git clone https://github.com/JDAI-CV/fast-reid.git

```

### 6. 其他库安装

```shell
pip install faiss-gpu
pip install emoji
pip inistall openpyxl
pip install loguru
pip install thop
pip install filterpy
pip install scikit-learn
pip install grad-cam==1.4.8
pip install timm

```

## 数据准备  

CrashEye开发了全球首套面向列车碰撞的基准，包括检测数据集`Crash2024`, `Crash-Seq`以及`Crash-ReID`。所有数据按照要求提供。

<div align="center">

| 数据类型 | 名称| 下载地址 | 存放地址 |
| ---------- | ---------- | ----------------------- | -------------------------------------------- |
| 检测| Crash2024 | [data](https://drive.google.com/drive/folders/1BJOdywj-hgXRKt_q0TEcBGpCV4Wojmhc?usp=drive_link) | **datasets** |
| 重识别 | Crash-ReID | [data](https://pan.baidu.com/s/17e5o7nZqMTBO0WxoDDfZvA?pwd=ks5f) | **datasets**|
| 跟踪 | Crash-Seq | [data]( https://pan.baidu.com/s/1FyOSl3A43Cibm6zxXlGYbA?pwd=gpju) | **datasets** |

</div>

## 模型库  
CrashEye在[configs](configs)中提供了``7``个案例，它们涵盖了纯openmmlab方法，纯ultralytics方法以及混合方法。也就是说，开发人员可以通过修``yaml``文件来搭建任意的模型。``需要注意的是，需要提供模型的预训练模型``:  
<div align="center">  

| 检测 | 跟踪 | 模型 | 配置文件 | 存放地址 |
| ---------- | ---------- | --------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| Centernet | MASORT | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/mix_crash_centernet_masort.yaml) | **pretrained/det** |
| YOLOv8 | Deepsort | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/mix_crash_yolov8s_deepsort.yaml) | **pretrained/det** |
| Centernet | Deepsort | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/mm_crash_centernet.yaml) |**pretrained/det** |
| Faster-RCNN | Deepsort | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/mm_crash_frcnn.yaml)  | **pretrained/det** |
| YOLOX | ByteTrack | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/mm_crash_yolox.yaml)  | **pretrained/det** |
| CVMR | MASORT | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew) | [config](configs/ult_crash_cvmrs.yaml)  | **pretrained/det** |
| YOLOv8s | MASORT | model[[baidu:csuw]](https://pan.baidu.com/s/1YjfNrMjzZW8y4461-lIKew)| [config](ult_crash_yolov8.yaml)  |  **pretrained/det** |

</div>
