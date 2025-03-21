FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive


##### Environment variables #####
ENV CUDA_HOME=/usr/local/cuda

##### Essential packages #####
RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    git \
    openssh-client \
    wget \
    vim \
    curl \
    build-essential \
    libvulkan1 \
    libssl-dev \
    pkg-config \
    python3-opencv \
    libopencv-dev \
    libpcl-dev \
    libglew-dev \
    libglvnd-dev
    

# install Cmake
ARG version=3.22
ARG build=1
RUN apt remove cmake -y
WORKDIR /tmp
RUN wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
RUN tar -xzvf cmake-$version.$build.tar.gz
WORKDIR /tmp/cmake-$version.$build
RUN ./bootstrap
RUN make -j8
RUN make install


##### SSH keys for GitHub #####
# ssh keys as build arguments
ARG ssh_prv_key
ARG ssh_pub_key

RUN mkdir -p -m 0700 /root/.ssh && \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts

RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub


##### Workspace setup #####
RUN mkdir -p /workspace
WORKDIR /workspace


##### sap_nerf #####
RUN git clone git@github.com:snt-arg/sap_nerf.git
WORKDIR /workspace/sap_nerf/
RUN pip install -r requirements.txt


##### PRENOM #####
# Pangolin
WORKDIR /workspace/
RUN apt-get install -y libepoxy-dev
RUN git clone https://github.com/stevenlovegrove/Pangolin.git
WORKDIR /workspace/Pangolin/
RUN git checkout aff6883c83f3fd7e8268a9715e84266c42e2efe3
RUN mkdir build && cd build && \
    cmake .. && \
    make -j && \
    make install
RUN ldconfig

# Eigen
WORKDIR /workspace/
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
RUN unzip eigen-3.4.0.zip
RUN mv eigen-3.4.0 eigen
WORKDIR /workspace/eigen/
RUN mkdir build && cd build && \
    cmake .. && \
    make install

# Multi-Object-NeRF
RUN apt-get update
WORKDIR /workspace
RUN git clone --recursive git@github.com:snt-arg/PRENOM.git

WORKDIR /workspace/PRENOM/dependencies/Multi-Object-NeRF/Core/third_party/tiny-cuda-nn
RUN git submodule update --init --recursive

WORKDIR /workspace/PRENOM/dependencies/Multi-Object-NeRF/Core/
RUN echo "Configuring and building Multi-Object-NeRF ..."
RUN cmake . -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -D TCNN_CUDA_ARCHITECTURES=70 -B build
RUN cmake --build build --config RelWithDebInfo --target all -- -j

WORKDIR /workspace/PRENOM/dependencies/Multi-Object-NeRF/
RUN echo "Configuring and building OfflineNeRF ..."
RUN cmake . -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -B build
RUN cmake --build build --config RelWithDebInfo --target all -- -j

# build final
WORKDIR /workspace/PRENOM/
RUN sh build.sh

# multi-objective optimization
WORKDIR /workspace/PRENOM/learntolearn
RUN pip install -r requirements.txt 

#### Clean up #####
# remove ssh keys
RUN rm -rf /root/.ssh