/**
 * @brief This file is part of PRENOM
 *
 * This file is based on a file from SQ-SLAM.
 *
 * Original work:
 * Copyright © 2015‑2017 Raúl Mur‑Artal, J.M.M. Montiel, and Juan D. Tardós 
 * (University of Zaragoza)
 * For more information, see <https://github.com/raulmur/ORB_SLAM2>
 * Copyright © 2022 Han, Xiao and Yang, Lu
 * For more information, see <https://github.com/XiaoHan-Git/SQ-SLAM>
 *
 * Modifications:
 * This file contains changes made by the Interdisciplinary Centre for Security, 
 * Reliability and Trust, University of Luxembourg in 2025.
 *
 * Copyright © 2025 Interdisciplinary Centre for Security, Reliability and Trust, 
 * University of Luxembourg.
 *
 * PRENOM is free software: you can redistribute it and/or modify it under the terms 
 * of the GNU General Public License as published by the Free Software Foundation, 
 * either version 3 of the License, or (at your option) any later version.
 *
 * PRENOM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * See the GNU General Public License version 3 for more details.
 *
 * You should have received a copy of the GNU General Public License version 3 
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "dependencies/Multi-Object-NeRF/Core/include/nerf_manager.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    //object-nerf-slam
    void SetNeRFManager(nerf::NerfManagerOnline* pNeRFManager);
    void InsertKeyFrameAndImg(KeyFrame *pKF,const cv::Mat& img,const cv::Mat& Instance, const cv::Mat& imgDepth);
    void NewDataToGPU();
    void UpdateObjNeRF();
    void GetUpdateBbox(Object_Map* pObj, vector<nerf::FrameIdAndBbox>& vFrameBbox);
    static float mfAngleChange;
    static int mnRestBetweenNeRFs;

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    // RO-MAP--------------------------------------
    //Update Object Size And Shape
    void UpdateObjSizeAndPose();

    //Merge Possible Objects
    void MergeObjects();

    //Merge Overlap Objects
    void MergeOverlapObjects();

    //t test 
    float tTest[101][4] = {0};

    std::mutex mNewDataNeRF;
    std::list<KeyFrame*> mlNewKeyFramesNeRF;
    std::list<cv::Mat> mlNewImgNeRF;
    std::list<cv::Mat> mlNewInstanceNeRF;
    std::list<cv::Mat> mlNewImgDepthNeRF;
    unsigned int mCurImgId = 0;
    std::vector<KeyFrame*> mvNeRFDataKeyFrames;

    nerf::NerfManagerOnline* mpNeRFManager;

    unsigned long int mnLastUpdateObjFrameId;

    set<Object_Map*> mvUpdateObj;

    //-----------------------------------------

    int mnKFsSinceLastNeRF = 0;
    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
