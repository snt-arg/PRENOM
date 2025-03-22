<div align="center">
    <h1>📇 PRENOM</h1>
    <p><i>Knowing objects on a first name basis</i></p>
    <a href="https://www.arxiv.org/abs/2503.01582">
    <img src="https://img.shields.io/badge/arXiv-2307.12815-b31b1b.svg" alt="arXiv">
  </a>

</div>

# Introduction  
PRENOM uses category-level meta-learned NeRF priors to accelerate object reconstruction and enable 4-DoF canoical object pose estimation. Priors are trained by Reptile meta-learning on numerous synthetic reconstruction tasks, while the NeRF architecture is also optimized per category using a multi-objective mixed-variable genetic algorithm. The system architecture can be seen below:

![System architecture](/docs/PRENOM/System_architecture_resized.png)

# Demos
All experiments performed on a laptop computer with only an [NVIDA T600 Laptop GPU](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/proviz-print-nvidia-T600-datasheet-us-nvidia-1670029-r5-web.pdf) with 4GB of memory.

### Synthetic Sequence
![Synthetic sequence](/docs/PRENOM/online_run.gif)

### Real Sequence
![Real sequence](/docs/PRENOM/real_online_run.gif)

### Object comparisons  
BASELINE is an RGB-D version of [RO-MAP](https://github.com/XiaoHan-Git/RO-MAP).  

![Object comparisons](/docs/PRENOM/objects.gif)

# License

This repo is GPLv3 Licensed (inherit ORB-SLAM2). The Multi-Object NeRF system based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (BSD 3-clause license). The implementation refers to [instant-ngp](https://github.com/NVlabs/instant-ngp) (Nvidia Source Code License-NC) and uses its marching cubes algorithm directly.

# Prerequisites

* [Eigen3](http://eigen.tuxfamily.org) (test version: 3.4.0)
* [OpenCV](http://opencv.org) (test version: 4.2.0)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (test version: 0.9)
* [CMake](https://cmake.org/) (test version: 3.28.3)
* [PCL](https://pointclouds.org/) (test version 1.10)

* [CUDA](https://developer.nvidia.com/cuda-toolkit) (test version: 12.2)

Test system: Ubuntu 0.04, GPU: NVIDIA T600

# Installation  
Clone the repository and it's dependencies
```
git clone --recursive git@github.com:snt-arg/PRENOM.git
```

Similar to RO-MAP, build first the Multi-Object-NeRF module and then the SLAM system
```
cd dependencies/Multi-Object-NeRF
sh build.sh <number-of-cores>
cd ../../
sh build.sh <number-of-cores>
```

# Datasets

# Running


# Acknowledgments

This repo is built on top of the **[RO-MAP](https://github.com/XiaoHan-Git/RO-MAP)** framework, and borrows a significant amount of their codebase. We thank the authors for their work. Other acknowledgements to: 

* SLAM: **[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)**, **[CubeSLAM](https://github.com/shichaoy/cube_slam)**, **[EAO-SLAM](https://github.com/yanmin-wu/EAO-SLAM)**

* NeRF: **[instant-ngp](https://github.com/NVlabs/instant-ngp)**, **[ngp_pl](https://github.com/kwea123/ngp_pl)**

# Cite

