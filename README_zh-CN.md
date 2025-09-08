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
- [许可证](#许可证)

## 简介
[CrashEye](https://github.com/YanJieWen/CrashEye)是第一个面向**列车耐撞性**设计的目标检测+跟踪项目。它可以在无需任何人工干预的情况下实现列车碰撞过程中的棋盘导航点的自主轨迹提取。

主分支代码目前在``Pytorch 2.0`` ``Python 3.8`` ``CUDA 11.8``版本上运行。  

- **集成多个检测/跟踪架构**

CrashEye集成的检测器包括[MMdetection](https://github.com/open-mmlab/mmdetection)/[ultralytics](https://github.com/ultralytics/ultralytics)/强基线[CVMR](https://github.com/YanJieWen/CVMR)。
跟踪器包括[MMTracking](https://github.com/open-mmlab/mmtracking)/强基线[MASORT](https://github.com/YanJieWen/MASORT)。另外，我们的强基线还包括一个自监督的Re-ID模块[LWTGPF](https://github.com/YanJieWen/LWTGPF-2025)。  


- **多场景适应性**

CrashEye在多源碰撞场景中是有效的，包括端部吸能碰撞，整车碰撞，缩比列车碰撞等。  
<div align="center">
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/6c29886f-ae7a-4a55-8be4-352ee85b7d3e"/>
</div>

- **速度快**

- **性能高**
