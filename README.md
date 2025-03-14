# PRENOM

**[Paper](https://www.arxiv.org/abs/2503.01582)** 

# License

This repo is GPLv3 Licensed (inherit ORB-SLAM2). The Multi-Object NeRF system based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (BSD 3-clause license). The implementation refers to [instant-ngp](https://github.com/NVlabs/instant-ngp) (Nvidia Source Code License-NC) and uses its marching cubes algorithm directly.

# Prerequisites

Our prerequisites include [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) requirements. If you encounter compilation problems, please refer to their issues first.

* [Eigen3](http://eigen.tuxfamily.org) (test version: 3.4.0)
* [OpenCV](http://opencv.org) (test version: 3.4.16)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (test version: 0.8)
* [CMake](https://cmake.org/) (test version: 3.25)

* [CUDA](https://developer.nvidia.com/cuda-toolkit) (test version: 11.8)

Test system: ubuntu (docker) 20.04, GPU: T600

# Acknowledgments

This repo is built on top of the **[RO-MAP](https://github.com/XiaoHan-Git/RO-MAP)** framework, and borrows a significant amount of their codebase. We thank the authors for their work.

* SLAM: **[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)**, **[CubeSLAM](https://github.com/shichaoy/cube_slam)**, **[EAO-SLAM](https://github.com/yanmin-wu/EAO-SLAM)**

* NeRF: **[instant-ngp](https://github.com/NVlabs/instant-ngp)**, **[ngp_pl](https://github.com/kwea123/ngp_pl)**

