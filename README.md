<div align="center">
    <h1>📇 PRENOM</h1>
    <p><i>🫂 Knowing objects on a first name basis 🫂</i></p>
    <a href="https://www.arxiv.org/abs/2503.01582">
    <img src="https://img.shields.io/badge/arXiv-2307.12815-b31b1b.svg" alt="arXiv">
  </a>

</div>

# Introduction  
PRENOM uses category-level meta-learned NeRF priors to accelerate object reconstruction and enable 4-DoF canoical object pose estimation. Priors are trained by Reptile meta-learning on numerous synthetic reconstruction tasks, while the NeRF architecture is also optimized per category using a multi-objective mixed-variable genetic algorithm. The system architecture can be seen below:

![System architecture](/docs/PRENOM/System_architecture_resized.png)

<!-- # Demos
All experiments performed on a laptop computer with only an [NVIDA T600 Laptop GPU](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/proviz-print-nvidia-T600-datasheet-us-nvidia-1670029-r5-web.pdf) with 4GB of memory.

### Synthetic Sequence
![Synthetic sequence](/docs/PRENOM/online_run.gif)

### Real Sequence
![Real sequence](/docs/PRENOM/real_online_run.gif) -->

### Object comparisons  
BASELINE is an RGB-D version of [RO-MAP](https://github.com/XiaoHan-Git/RO-MAP).  

![Object comparisons](/docs/PRENOM/objects.gif)

# License

This repo is GPLv3 Licensed (inherit ORB-SLAM2). The Multi-Object NeRF system based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (BSD 3-clause license). The implementation refers to [instant-ngp](https://github.com/NVlabs/instant-ngp) (Nvidia Source Code License-NC) and uses its marching cubes algorithm directly.

# Prerequisites

* [Eigen3](http://eigen.tuxfamily.org) (test version: 3.4.0)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) (test version: v0.9)
* [CMake](https://cmake.org/) (test version: 3.22.1)
* [PCL](https://pointclouds.org/) (test version 1.10)

Tested on: 
* Ubuntu 20.04 LTS, GPU: T600 Laptop GPU (Nvidia driver: 535), CUDA 12.2, OpenCV 4.2.0
* Ubuntu 24.04 LTS, GPU: RTX 5090 (Nvidia driver: 570), CUDA 12.8, OpenCV 4.6.0

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
The six synthetic sequences (from S0 to S5 in the paper) and the two real sequences (scene1 and scene2 from RO-MAP) can be downloaded from [here](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/saad_ejaz_uni_lu/EmyhDvV4eBJAgq74EzDMlt8BoUR8NowcpOXPU-A05GPakQ?e=IIc1Dd). S0 corresponds to the [Cube Diorama Dataset](https://github.com/jc211/nerf-cube-diorama-dataset), albiet with config modified to work with PRENOM. The real sequences are the ones provided by RO-MAP but also with the configuration modified.

# Available Priors  
The priors for 9 categories are available in the `cookbook` folder, with their YOLO class IDs as the folder name. 
| Category name   | YOLO category name     | ID         |
| --------------- | ---------------------- | -----------|
| ball            | sports ball            | 32         |
| mug             | cup                    | 41         |
| chair           | chair                  | 56         |
| plant           | potted plant           | 58         |
| display         | tv                     | 62         |
| laptop          | laptop                 | 63         |
| mouse           | mouse                  | 64         |
| keyboard        | keyboard               | 66         |
| book            | book                   | 73         |

Each folder consists of:
* `weights.json` - the meta-learned initial NeRF parameters.
* `model.ply` - the normalized mesh used to align the prior with the actual object.
* `density.ply` - the prior density grid used to align priors and for probabilistic ray sampling.
* `network.json` - the category-specific architecture that was optimised.
* `config.json` - category-specific object association configuration.

The default configurations can be found in `cookbook/0/`, and `cookbook/recipes.txt` lists all the known categories along with their class IDs.

**Details on how to train a new prior are in the [README](https://github.com/snt-arg/PRENOM/blob/main/learntolearn/README.md) of the `learntolearn` folder.**

# Running
To run, navigate to the PRENOM directory and run using: 
```
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./examples/RGBD/rgbd_tum ./vocabulary/ORBvoc.bin ./dependencies/Multi-Object-NeRF/Core/configs/base.json <path/to/dataset/S1>
```
You can also run a monocular version by using the target `./examples/Monocular/mono_tum`. However, monocular data association is inherently noisy and the run has to be repeated until satisfactory results, as also mentioned by RO-MAP.

# Known Issues  
1. **`struct cudaPointerAttributes has no member named 'memoryType'` while building PRENOM.**  
    This can happen since we are using an older version of Pangolin, and in later versions of CUDA, this attribute was renamed to `type`. To fix this, just rename `.memoryType` to `.type` in the file `/usr/local/include/pangolin/image/memcpy.h` (or equivalent location of your Pangolin installation) and re-build PRENOM. 

# Acknowledgments

This repo is built on top of the **[RO-MAP](https://github.com/XiaoHan-Git/RO-MAP)** framework, and borrows a significant amount of their codebase. We thank the authors for their work. Other acknowledgements to: 

* SLAM: **[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)**, **[CubeSLAM](https://github.com/shichaoy/cube_slam)**, **[EAO-SLAM](https://github.com/yanmin-wu/EAO-SLAM)**
* NeRF: **[instant-ngp](https://github.com/NVlabs/instant-ngp)**, **[ngp_pl](https://github.com/kwea123/ngp_pl)**
* Farthest Point Sampling: **[FPS](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU)**

# Cite

