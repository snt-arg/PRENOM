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


#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "nerf_model.h"
#include "nerf_manager.h"
#include <fstream>

namespace nerf{

NerfManagerOffline::NerfManagerOffline(const string datasetPath, const string networkConfigFile, bool useDenseDepth)
{
    msDatasetPath = datasetPath;
    msNetworkConfigFile = networkConfigFile;
    mbUseDenseDepth = useDenseDepth;
}

bool NerfManagerOffline::Init()
{
    //Detect the number of GPUs
    cudaGetDeviceCount(&mNumGPU);
    cout<<"mNumGPU: "<<mNumGPU<<endl;
    if(mNumGPU < 1)
    {
        cerr << "Can not Detect GPU ... "<<endl;
        exit(0);
    }

    NeRF::GPUnum = mNumGPU;
    cout<<"Detect "<<mNumGPU<<" GPU ..."<<endl;

    return true;
}

bool NerfManagerOffline::ReadDataset()
{
    //Create a Dataset for each GPU
    std::vector<std::thread> threads_data;
    for(int i=0;i<mNumGPU;i++)
    {
        std::shared_ptr<NeRF_Dataset> pDataset = std::make_shared<NeRF_Dataset>(mbUseDenseDepth);
        mvpDataset.push_back(pDataset);
        pDataset->mGPUid = i;
        if(!pDataset->ReadDataset(msDatasetPath))
        {
            cerr << "Read Train Data error..."<<endl;
            exit(0);
        }
        threads_data.emplace_back(std::thread(&NeRF_Dataset::DataToGPU,pDataset));
    }

    for(auto &thread: threads_data) 
        thread.join();

    cudaDeviceSynchronize();
    return true;
}

bool NerfManagerOffline::CreateNeRF(const string objectFile, const json &systemConfig)
{
    //test file
    std::ifstream file(objectFile);
    if(!file)
    {
        cerr << "object file error..."<<endl;
        return false;
    }
    
    //Create NeRF Instance
    std::shared_ptr<NeRF> NeRFInstance = std::make_shared<NeRF>(
        systemConfig["save_model"],
        systemConfig["save_identifier"],
        systemConfig["visualize"],
        systemConfig["output_dir"]);
    mvpNeRFs.push_back(NeRFInstance);
    
    //Associate Dataset
    NeRFInstance->mpTrainData = mvpDataset[NeRFInstance->mGPUid];

    //Create network 
    if(!NeRFInstance->CreateModelOffline(objectFile,
                                         msNetworkConfigFile,
                                         mbUseDenseDepth,
                                         systemConfig["do_meta"],
                                         systemConfig["load_model"],
                                         systemConfig["load_path"]))
    {
        cerr << "Create NeRF error ..." <<endl;
        exit(0);
    }
    
    //Training
    if (systemConfig["do_meta"])
        mvThreads.emplace_back(std::thread(&NeRF::TrainMeta,
                                            NeRFInstance,
                                            systemConfig["meta_n_loops"],
                                            systemConfig["meta_n_steps"],
                                            systemConfig["meta_n_iters_per_step"]));
    else
        mvThreads.emplace_back(std::thread(&NeRF::TrainOffline,
                                            NeRFInstance,
                                            systemConfig["n_steps"],
                                            systemConfig["n_iters_per_step"]));
    return true;
    
}

bool NerfManagerOffline::WaitThreadsEnd()
{
    if(mvThreads.empty())
        return false;

    for(std::thread& thread : mvThreads)
        thread.join();
    return true;
}

std::shared_ptr<NeRF> NerfManagerOffline::GetNeRF(int idx)
{
    if(idx >= mvpNeRFs.size())
    {
        cerr << "NeRF Idx error ... "<<endl;
        exit(0);
    }
    return mvpNeRFs[idx];
}

vector<std::shared_ptr<NeRF>> NerfManagerOffline::GetAllNeRF()
{
    return mvpNeRFs;
}

vector<Eigen::Matrix4f> NerfManagerOffline::GetAllTwc()
{
    return mvpDataset[0]->mvIamgesPose;
}

void NerfManagerOffline::GetIntrinsics(float& fx,float& fy,float& cx,float& cy)
{
    fx = mvpDataset[0]->fx;
    fy = mvpDataset[0]->fy;
    cx = mvpDataset[0]->cx;
    cy = mvpDataset[0]->cy;
}


//NerfManagerOnline
NerfManagerOnline::NerfManagerOnline(const string network_config_file,bool UseSparseDepth, int TrainStepIterations)
    : mNetworkConfigFile(network_config_file), mbUseSparseDepth(UseSparseDepth), mnTrainStepIterations(TrainStepIterations){}

bool NerfManagerOnline::Init()
{
    cudaGetDeviceCount(&mNumGPU);
    cout<<endl<<"mNumGPU: "<<mNumGPU<<endl;
    if(mNumGPU < 1)
    {
        cerr << "Can not Detect GPU ... "<<endl;
        exit(0);
    }

    NeRF::GPUnum = mNumGPU;
    cout<<"Detect "<<mNumGPU<<" GPU ..."<<endl;
    
    // //Read config
    // if(!NeRF_Model::ReadNetworkConfig(mNetworkConfigFile))
    // {   
    //     cerr << "Read Network Config error..."<<endl;
    //     exit(0);
    // }

    return true;
}

void NerfManagerOnline::DatasetInit(float fx,float fy,float cx,float cy,int H,int W, size_t imgs)
{
    //Dataset
    //std::vector<std::thread> threads_data;
    for(int i=0;i<mNumGPU;i++)
    {   
        std::shared_ptr<NeRF_Dataset> pDataset = std::make_shared<NeRF_Dataset>(mbUseSparseDepth);
        mvpDataset.push_back(pDataset);
    
        pDataset->mGPUid = i;
        pDataset->fx = fx;
        pDataset->fy = fy;
        pDataset->cx = cx;
        pDataset->cy = cy;
        pDataset->H = H;
        pDataset->W = W;
        pDataset->mfDepthScale = 1.0f;
        pDataset->mnImages = imgs;

        //threads_data.emplace_back(std::thread(&NeRF_Dataset::InitDataToGPU,pDataset));
        pDataset->InitDataToGPU();
    }

    //for(auto &thread: threads_data) 
    //    thread.join();
    cudaDeviceSynchronize();

}

void NerfManagerOnline::NewFrameToDataset(unsigned int imgId,const string timestamp, cv::Mat& img, cv::Mat& instance, const cv::Mat& depth_img, const Eigen::Matrix4f& pose)
{

    if(mNumGPU < 2)
    {
        //one GPU
        std::shared_ptr<NeRF_Dataset> pDataset = mvpDataset[0];
        pDataset->Temp_Img = img;
        pDataset->Temp_Instance = instance;
        pDataset->Temp_Depth = depth_img;
        pDataset->Temp_Pose = pose;
        pDataset->FrameDataToGPU(imgId,timestamp);
    }
    else
    {   //Multiple GPUs
        std::vector<std::thread> threads_data;
        for(int i=0;i<mNumGPU;i++)
        {
            std::shared_ptr<NeRF_Dataset> pDataset = mvpDataset[i];
            pDataset->Temp_Img = img;
            pDataset->Temp_Instance = instance;
            pDataset->Temp_Depth = depth_img;
            pDataset->Temp_Pose = pose;
            threads_data.emplace_back(std::thread(&NeRF_Dataset::FrameDataToGPU,pDataset,imgId,timestamp));
        }

        for(auto &thread: threads_data) 
            thread.join();
    }  
}

void NerfManagerOnline::UpdateDataset(unsigned int CurId,unsigned int FrameNum,const vector<Eigen::Matrix4f>& Poses)
{
    std::vector<std::thread> threads_data;
    for(int i=0;i<mNumGPU;i++)
    {
        std::shared_ptr<NeRF_Dataset> pDataset = mvpDataset[i];
        pDataset->mvTemp_Update_Pose.clear();
        pDataset->mvTemp_Update_Pose = Poses;
        
        threads_data.emplace_back(std::thread(&NeRF_Dataset::UpdateDataGPU,pDataset,CurId,FrameNum));
    }

    for(auto &thread: threads_data) 
        thread.join();

}

size_t NerfManagerOnline::CreateNeRF(const int classId, const Eigen::Matrix4f &ObjTow, const nerf::BoundingBox &BoundingBox, const bool known)
{

    std::shared_ptr<NeRF> NeRFInstance = std::make_shared<NeRF>();
    size_t idx = mvpNeRFs.size();
    mvpNeRFs.push_back(NeRFInstance);

    NeRFInstance->mpTrainData = mvpDataset[NeRFInstance->mGPUid];
    size_t data_mutex_idx = NeRFInstance->mpTrainData->mvUpdateMutex.size();
    NeRFInstance->mDataMutexIdx = data_mutex_idx;
    NeRFInstance->mpTrainData->mvUpdateMutex.emplace_back(new std::mutex());
    
    NeRFInstance->SetAttributes(classId,ObjTow,BoundingBox,mvpDataset[0]->mnImages);
    if(!NeRFInstance->CreateModelOnline(mbUseSparseDepth,mnTrainStepIterations,classId,known))
    {
        cerr << "Create NeRF error ..." <<endl;
        exit(0);
    }
    
    //We create a thread per object instead of using a thread pool. This facilitates debugging and understanding. 
    //If you are faced with many objects (creating many threads will affect the system operation), we recommend that
    //you modify the training logic of the objects and use the thread pool to improve performance.
    mvThreads.emplace_back(std::thread(&NeRF::TrainOnline,NeRFInstance));
    return idx;
}

bool NerfManagerOnline::WaitThreadsEnd()
{
    if(mvThreads.empty())
        return false;
    
    for(std::shared_ptr<NeRF> pNeRF : mvpNeRFs)
    {
        pNeRF->RequestFinish();
    }

    for(std::thread& thread : mvThreads)
        thread.join();

    cout<<"All NeRF threads completed ..."<<endl;
    return true;
}

void NerfManagerOnline::RenderNeRFsTest(const string out_path,const size_t Idx,const vector<string>& timestamp, const vector<FrameIdAndBbox>& vBbox, const vector<Eigen::Matrix4f>& vTwc,const float radius)
{
    if(mvpNeRFs.empty())
        return;
    mvpNeRFs[Idx]->RenderTestImg(out_path,timestamp,vTwc,vBbox,radius);
}


int NerfManagerOnline::GetFrameIdx(double timastamp)
{
    string stamp = to_string(timastamp);
    int idx = -1;
    if(mvpDataset[0]->mStampToIdx.find(stamp) != mvpDataset[0]->mStampToIdx.end())
        idx = int(mvpDataset[0]->mStampToIdx[to_string(timastamp)]);

    return idx;
}

void NerfManagerOnline::UpdateNeRFBbox(const size_t idx, const vector<nerf::FrameIdAndBbox>& vFrameBbox,const int train_step)
{
    if(vFrameBbox.empty())
        return;
    mvpNeRFs[idx]->UpdateFrameBBox(vFrameBbox,train_step);
}

void NerfManagerOnline::DrawMesh(size_t idx)
{
    if(idx > (mvpNeRFs.size() - 1))
        return;
    
    mvpNeRFs[idx]->DrawCPUMesh();
    //mvpNeRFs[idx]->DrawMesh();
}

}
