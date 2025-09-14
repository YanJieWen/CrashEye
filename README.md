<div align="center">
  <img src="assets/CrashEye-logo-removebg-preview.png" width="400"/>
  <div>&nbsp;</div>
</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## 📖 Contents
- [Introduction](#Introduction)
- [News](#News)
- [Installation](#Installation)
- [Dataset](#Dataset)
- [Model](#model)
- [Usage](#usage)
- [Demo](#Demo)
- [Notes](#Notes)
- [Acknowledgements](#Acknowledgements)
- [License](#License)

## Introduction
[CrashEye](https://github.com/YanJieWen/CrashEye) is an intelligent perception and trajectory interpolation framework for train collision sequence images.  
It aims to bridge the gap between real-world train collision experiments and finite element simulation models, enabling accurate reproduction and prediction of collision dynamics.  
The master branch code currently supports on `Pytorch 2.0` `Python 3.8` and `CUDA11.8`. 

**🔥Hightlights**
- **Integrating Multiple Detection/Tracking Architectures**:

 CrashEye integrates detectors including [MMdetection](https://github.com/open-mmlab/mmdetection)/[ultralytics](https://github.com/ultralytics/ultralytics)/strong baseline [CVMR](https://github.com/YanJieWen/CVMR). Trackers include [MMTracking](https://github.com/open-mmlab/mmtracking)) and a strong baseline [MASORT](https://github.com/YanJieWen/MASORT). Furthermore, our strong baseline includes a self-supervised Re-ID module [LWTGPF](https://github.com/YanJieWen/LWTGPF-2025). Developers can combine several components and modules to create customized models.
- **Multi-scenes adaptability** 

CrashEye is effective in multi-source collision scenarios, including absorber crash, full-train crash, and scaled train collisions.
<div align="center">
  
<img src="assets/trackres.gif" width="600"/>

</div>  

- **High efficiency**

All operations (detection/tracking) are performed on a powerful GPU (RTX4090@24G).
Leveraging the existing data processing pipeline, there is no need to directly process the high-resolution images captured by the high-speed shot camera (HS2C); instead, memory consumption is significantly reduced through image downscaling.
<div align="center">  

<img src="assets/data_pipeline.png" />   

</div>


- **High performance**

CrashEye includes a strong baseline model that integrates [CVMR](https://github.com/YanJieWen/CVMR) for small object detection, [LWTGPF](https://github.com/YanJieWen/LWTGPF-2025) for deep   instance appearance feature extraction, and [MASORT](https://github.com/YanJieWen/MASORT) for adaptive cross-frame tracking. The performance of each baseline is reported in correspond repository. 

## News  
- **⭐2025-09-08**

  - Initial release of CrashEye, supporting autonomous perception of train collision sequence images.

  - Released dataset interfaces: Crash2024, Crash-ReID, and Crash-Seq.

  - Provided pretrained weights for various models, including detection and Re-ID.

- **⭐2025-09-14**
  - Release the English version of the [instructions](README.md)

## Installation 

### 1. CrashEye

``` shell
git clone  https://github.com/YanJieWen/CrashEye.git
cd CrashEye
python setup.py develop
```
### 2. [openmmlab](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html) baslines  

``` shell
pip install -U openmim
mim install mmengine
```

- **Install mmcv**

Version of mmdet and mmcv shoudle be aligen, [Reference](https://github.com/open-mmlab/mmtracking/blob/master/docs/zh_cn/install.md)。Installing [mmcv](https://github.com/open-mmlab/mmcv) through wheel, Refernce list is [here](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)  
[![mmcv](https://img.shields.io/badge/mmcv-1.7.2-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing)
```shell
shell mim install mmcv_full-1.7.2-cp38-cp38-manylinux1_x86_64.whl
```
- **Install mmdetection**

```shell
cd modeling
cd mmdetection
%git checkout tags/v2.28.0
pip install -v -e .
python setup.py install
cd ..
```
- **Install mmtracking**

```shell
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking git checkout tags/v0.14.0
pip install -v -e .
python setup.py install
cd ..
```

### 3. Ultralytics  

- **Install CVMR**
```shell
git clone https://github.com/YanJieWen/CVMR.git
cd CVMR
```

- **Install [Casual Conv1D](https://github.com/Dao-AILab/causal-conv1d/releases) and [SSM](https://github.com/state-spaces/mamba/releases) operataions**

[![Casual Conv1D](https://img.shields.io/badge/CNN-cuda-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing) [![SSM](https://img.shields.io/badge/Mamba-cuda-blue)](https://drive.google.com/drive/folders/1pAr4dmMDkEW2Wvl4af2sknU2GPN2856S?usp=sharing)  

```shell
shell pip install causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl pip install --upgrade pip pip install -v -e .
cd ..
```

- **Change code**

Add the forced half-precision comment in ``line 113`` of ``modeling/CVMR/ultralytics/engine/validator.py`` (training may result in Nan under certain conditions).
