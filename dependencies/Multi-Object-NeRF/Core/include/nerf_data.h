/**
 * @brief This file is part of PRENOM
 *
 * This file is based on a file from RO-MAP which refers to instant-ngp.
 *
 * Original work:
 * Copyright © 2022, NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights 
 * in and to this software, related documentation, and any modifications thereto. Any use, 
 * reproduction, disclosure, or distribution of this software and related documentation 
 * without an express license agreement from NVIDIA CORPORATION is strictly prohibited.
 * 
 * Copyright © 2023 Han, Xiao and Liu, Houxuan and Ding, Yunchao and Yang, Lu
 * For more information, see <https://github.com/XiaoHan-Git/RO-MAP>
 *
 * Modification:
 * This file contains changes made by the Interdisciplinary Centre for Security, 
 * Reliability and Trust, University of Luxembourg.
 *
 * Copyright © 2025 Interdisciplinary Centre for Security, Reliability and Trust, 
 * University of Luxembourg.
 *
 * All changes remain licensed under the NVIDIA Source Code License-NC.
 * See the full license for details.
 */


#pragma once
#include <tiny-cuda-nn/common_device.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <mutex>
#include <opencv2/opencv.hpp>
using namespace std;

namespace nerf{
    
struct MetaData
{
    const float* pixels;
	const float* depth;
	const uint8_t* instance;
    const Eigen::Matrix4f* Pose;
};

class NeRF_Dataset 
{
public:
    NeRF_Dataset(bool useDepth) : mbUseDepth(useDepth){};
    ~NeRF_Dataset();

    //offline
    bool ReadDataset(const string path);
    bool DataToGPU();
    
    //online
    bool InitDataToGPU();
    void FrameDataToGPU(unsigned int imgId,const string timestamp);
    //Update the frame pose passed in before, not used
    void UpdateDataGPU(unsigned int CurId,unsigned int FrameNum);

    //offline
    size_t mnImages = 0;
    vector<string> mvImagesPath;
	vector<string> mvDepthsPath;
	vector<string> mvInstancesPath;
    vector<Eigen::Matrix4f> mvIamgesPose;

    //Online
    //Temporary image data
    cv::Mat Temp_Img;
    cv::Mat Temp_Instance;
    cv::Mat Temp_Depth;
    Eigen::Matrix4f Temp_Pose;
    uint32_t mFrameDataNum = 0;
    vector<Eigen::Matrix4f> mvIamgesPose_online;
    vector<Eigen::Matrix4f> mvTemp_Update_Pose;
    vector<std::mutex*> mvUpdateMutex;

    float fx;
    float fy;
    float cx;
    float cy;
    int H;
    int W;
    bool mbUseDepth = false;
	float mfDepthScale;

    //GPUid
    int mGPUid;
    std::map<string,uint32_t> mStampToIdx;
    //data on GPU
    vector<tcnn::GPUMemory<float>> mvPixelMemory;
    vector<tcnn::GPUMemory<float>> mvDepthMemory;
	vector<tcnn::GPUMemory<uint8_t>> mvInstanceMemory;
    tcnn::GPUMemory<Eigen::Matrix4f> mPosesMemory;
    tcnn::GPUMemory<MetaData> mMetadataMemory;
    tcnn::GPUMemory<float> mIntrinsicsMemory;

};
}