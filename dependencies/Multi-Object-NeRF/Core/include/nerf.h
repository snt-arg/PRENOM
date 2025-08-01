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
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#include "common.h"
#include <condition_variable>


using namespace std;

namespace nerf{

class NeRF_Model;
class NeRF_Dataset;

class NeRF
{
public:

    NeRF(const bool saveModel = true, const int saveIdentifier = 999, const bool visualize = true, const std::string outputDir = "./output/");

    //offline train function
    bool CreateModelOffline(const string path, const string networkPath, const bool useDenseDepth, const bool doMeta, const bool loadModel, const string modelPath);
    bool ReadBboxOffline(const string path);
    void TrainOffline(const int nSteps, const int nItersPerStep);
    void TrainMeta(const int nMetaLoops, const int nMetaSteps, const int nMetaItersPerStep);
    
    //online train function
    void SetAttributes(const int Class,const Eigen::Matrix4f& ObjTow,const BoundingBox& BoundingBox,size_t numBbox);
    bool CreateModelOnline(bool useSparseDepth, int Iterations, const int classId, const bool known = true);
    void TrainOnline();
    void UpdateFrameBBox(const vector<nerf::FrameIdAndBbox>& vFrameBbox, const int train_step);
    bool CheckFinish();
    void RequestFinish();

    //tools functions
    void RenderTestImg(const string out_path, const vector<string>& timestamp,const vector<Eigen::Matrix4f>& testTwc, const vector<FrameIdAndBbox>& testBbox, const float radius);
    inline int GetTrainIters(int trainStep);
    vector<Eigen::Matrix4f> GetTwc();
    BoundingBox GetBoundingBox();
    Eigen::Matrix4f GetObjTow();
    CPUMeshData& GetCPUMeshData();
    vector<FrameIdAndBbox> GetFrameIdAndBBox();
    void DrawCPUMesh();
    void DrawMesh();

    //NeRF id
    static int curId;
    int mId;
    //CUDA GPU
    static int GPUnum;
    static int curGPUid;
    int mGPUid =-1;

    //3D Bbox
    int mClass;
    uint8_t mInstanceId;
    Eigen::Matrix4f mObjTow;
    BoundingBox mBoundingBox; 

    std::vector<FrameIdAndBbox> mFrameIdBbox;
    size_t mnBbox;
    std::mutex mUpdateBbox;
    std::condition_variable mCond;

    std::mutex mFinishMutex;
    bool mbFinishRequested = false;
    
    bool mbUseDepth;
    int mnIteration;
    bool mbVisualize = false;
    bool mbSave = false; // one bool to save both mesh (meta + simple) and model (only meta)
    int mnIdentifier;
    std::string mOutputDir;

    //train data
    std::shared_ptr<NeRF_Dataset> mpTrainData;
    size_t mDataMutexIdx;

    //train model
    std::shared_ptr<NeRF_Model> mpModel;

    int mnTrainStep = 0;
   
    //CPU Mesh
    CPUMeshData mCPUMeshData;

    //Mesh
    MeshData mMeshData;

};
}