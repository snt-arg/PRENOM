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

#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
namespace ORB_SLAM2
{

System::System(const string &strVocFile, const string &strSettingsFile,const string &strNeRFConfigFile, const string &strDataset, int nimgs, const eSensor sensor,
     const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad;
    if(strVocFile.substr(strVocFile.length()-3,strVocFile.length()) == "bin")
        bVocLoad =  mpVocabulary->loadFromBinaryFile(strVocFile);
    else
        bVocLoad =  mpVocabulary->loadFromTextFile(strVocFile);

    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);
    
    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor, strDataset);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    // Initialize the Semantics Manager thread and launch
    mpObjectManager = new ObjectManager(mpMap, strDataset, strSettingsFile);
    mptObjectManager = new thread(&ObjectManager::Run, mpObjectManager);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);
    mpTracker->SetObjectManager(mpObjectManager);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    mpObjectManager->SetLocalMapper(mpLocalMapper);


    //RO-MAP
    /*
    Due to the assumption that our objects are placed on a horizontal plane,
    the X-Y plane of the world coordinate system is required to be parallel to the ground plane,
    so we read the ground truth of the initial frame to align.
    Or keep the camera as horizontal as possible while the system is initializing.
    NOTE: This is easy to do with an IMU. But our focus is not here.
    */
    mpTracker->LoadGroundtruthPose(strDataset);

    int UseSparseDepth = fsSettings["NeRF.UseSparseDepth"];
    int TrainStepIterations = fsSettings["NeRF.TrainStepIterations"];
    mpNeRFManager = new nerf::NerfManagerOnline(strNeRFConfigFile,UseSparseDepth,TrainStepIterations); 
    mpNeRFManager->Init();

    mpLocalMapper->SetNeRFManager(mpNeRFManager);
    mpMapDrawer->SetNeRFManager(mpNeRFManager);

    float fx = fsSettings["Camera.fx"];
    float fy = fsSettings["Camera.fy"];
    float cx = fsSettings["Camera.cx"];
    float cy = fsSettings["Camera.cy"];
    int H = fsSettings["Camera.H"];
    int W = fsSettings["Camera.W"];
    
    //Please note that memory is allocated in the GPU directly according to the total number of images.
    //The ideal online operation should allocate memory dynamically and incrementally.
    mpNeRFManager->DatasetInit(fx,fy,cx,cy,H,W,nimgs);
    
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const cv::Mat &imgInstance, const double &timestamp, const string &strDatasetPath)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,imgInstance,timestamp,strDatasetPath);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const cv::Mat &ImgInstance, const double &timestamp, const string &strDatasetPath)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,ImgInstance,timestamp,strDatasetPath);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    long int Sumtime = 0;
    for(auto time : mpTracker->Asstime)
    {
        Sumtime += time;
    }
    cout<<"Ass average time: "<< Sumtime / mpTracker->Asstime.size() <<endl;

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

    /* if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    } */

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.

    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose() /* *Two */;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}


//save objects
void System::SaveObjects(const string &filename)
{
    cout << endl << "Saving objects to " << filename << " ..." << endl;
    
    vector<Object_Map*> vpObjs = mpMap->GetAllObjectMap();
    sort(vpObjs.begin(),vpObjs.end(),[](Object_Map* p1,Object_Map* p2){return p1->mnCreatFrameId < p2->mnCreatFrameId;});
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    
    int i=0;
    g2o::SE3Quat Twobj;
    Eigen::Quaterniond R;
    Eigen::Vector3d t;

    f<<"# ID  class  tx  ty  tz  qx  qy  qz  qw  a1  a2  a3"<<endl;
    for(Object_Map* pObj : vpObjs)
    {
        if(pObj->IsBad())
            continue;
        if(pObj->mnObs >  20)
        {   
            cout <<"mnClass: "<< pObj->mnClass << "   mnObs: " <<pObj->mnObs<< endl;
            Twobj = pObj->mShape.mTobjw.inverse();
            R = Twobj.rotation();
            t = Twobj.translation();
            f<< i << " "<<pObj->mnClass<<" "<<t(0)<<" "<<t(1)<<" "<<t(2)<<" "<<R.x()<<" "<< R.y()<<" "<< R.z()<<" "<<R.w()<<" ";
            f<< pObj->mShape.a1 << " "<<pObj->mShape.a2 <<" "<<pObj->mShape.a3<<endl;
            i++;
        }
    }

    f.close();
    cout << endl << "Objects saved!" << endl;
}

void System::FinishNeRFs()
{
    mpNeRFManager->WaitThreadsEnd();
}

void System::RenderNeRFsTest(const string out_path)
{
    vector<Object_Map*> vpObjs = mpMap->GetAllObjectMap();
    sort(vpObjs.begin(),vpObjs.end(),[](Object_Map* p1,Object_Map* p2){return p1->mnCreatFrameId < p2->mnCreatFrameId;});
    
    for(Object_Map* pObj : vpObjs)
    {
        if(pObj->IsBad())
            continue;
        if(!pObj->haveNeRF)
            continue;
        
        vector<string> timestamp;
        vector<nerf::FrameIdAndBbox> Bboxes;
        vector<Eigen::Matrix4f> vTwc;
        for(auto& it : pObj->mHistoryBbox)
        {   
            double stamp = it.first;
            if(pObj->mKeyFrameHistoryBbox.count(stamp))
            {
                continue;
            }
            timestamp.push_back(to_string(stamp));

            Bbox box = it.second;
            nerf::FrameIdAndBbox a;
            a.x = box.x;
            a.y = box.y;
            a.w = box.width;
            a.h = box.height;
            Bboxes.push_back(a);
            vTwc.push_back(pObj->mHistoryTwc[stamp]);

        }
        
        if(timestamp.empty())
            continue;
        
        size_t idx = pObj->pNeRFIdx;
        float RenderRadius = pObj->mShape.mfMaxDist * 5;
        mpNeRFManager->RenderNeRFsTest(out_path,idx,timestamp,Bboxes,vTwc,RenderRadius);

    }

/* [TODO] - add update to KeyFrame poses
//-------------------------------------------------------------------
    //if update old dataset, use the following code
    map<string,Eigen::Matrix4f> vFrameTwc;
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose();
        
        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
        Eigen::Matrix3f eRwc = Converter::toMatrix3d(Rwc).cast<float>();
        Eigen::Vector3f etwc = Converter::toVector3d(twc).cast<float>();
        Eigen::Matrix4f eTwc = Eigen::Matrix4f::Identity();
        eTwc.topLeftCorner(3,3) = eRwc;
        eTwc.col(3).head<3>() = etwc;
        string stamp = to_string(*lT);
        cout << stamp <<endl;
        cout << eTwc <<endl;
        vFrameTwc.insert(std::pair<string,Eigen::Matrix4f>(stamp,eTwc));
        //vFrameTwc[stamp] = eTwc;
    }
    
    vector<Object_Map*> vpObjs = mpMap->GetAllObjectMap();
    sort(vpObjs.begin(),vpObjs.end(),[](Object_Map* p1,Object_Map* p2){return p1->mnCreatFrameId < p2->mnCreatFrameId;});
    
    for(Object_Map* pObj : vpObjs)
    {
        if(pObj->IsBad())
            continue;
        if(!pObj->haveNeRF)
            continue;
        
        vector<string> timestamp;
        vector<nerf::FrameIdAndBbox> Bboxes;
        vector<Eigen::Matrix4f> vTwc;
        for(auto& it : pObj->mHistoryBbox)
        {   
            double stamp = it.first;
            if(pObj->mKeyFrameHistoryBbox.count(stamp))
            {
                continue;
            }

            if(vFrameTwc.find(to_string(stamp)) == vFrameTwc.end())
            {
                continue;
            }
            
            timestamp.push_back(to_string(stamp));

            Bbox box = it.second;
            nerf::FrameIdAndBbox a;
            a.x = box.x;
            a.y = box.y;
            a.w = box.width;
            a.h = box.height;
            Bboxes.push_back(a);
            vTwc.push_back(vFrameTwc[to_string(stamp)]);
            
        }
        
        if(timestamp.empty())
            continue;
        size_t idx = pObj->pNeRFIdx;
        mpNeRFManager->RenderNeRFsTest(out_path,idx,timestamp,Bboxes,vTwc);

    }
//------------------------------------------------------------------- 
*/
}

} //namespace ORB_SLAM
