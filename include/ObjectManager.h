/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/18/2022
* Author: Xiao Han
*/
#ifndef OBJECTMANAGER_H
#define OBJECTMANAGER_H

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include "third-party/pcl_custom/WarpPointRigid4D.h"
#include "dependencies/line_lbd/include/line_lbd_allclass.h"

#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "ObjectFrame.h"
#include "Converter.h"
#include "LocalMapping.h"
#include "dependencies/g2o/g2o/types/se3quat.h"
#include "Frame.h"
#include <vector>
#include <map>
#include <mutex>


using namespace g2o;
using namespace std;

namespace ORB_SLAM2
{
class Frame;
class LocalMapping;
class MapPoint;
class Map;
class Object_Map;
class KeyFrame;

class ObjectManager
{
public:

    struct Task
    {
        Frame* frame;
        KeyFrame* keyframe;
        cv::Mat imgColor;
        cv::Mat imgGray;
        cv::Mat imgInstance;
        cv::Mat imgDepth;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    };

    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    ObjectManager(Map* pMap, const string &strDataset, const string &strSettingPath);
    void AddTaskToQueue(Task task);
    void Run();
    void SetLocalMapper(LocalMapping* pLocalMapper);

    // points to draw
    vector<cv::Point> mvPointsToDrawer;

    
protected:

    // Map
    Map* mpMap;

    // Local Mapper
    LocalMapping* mpLocalMapper;

    // frame queue
    std::queue<Task> mTaskQueue;

    // parameters from Tracking thread
    int mnImgWidth;
    int mnImgHeight;
    cv::Mat mDistCoef;
    cv::Mat mK;
    Eigen::Matrix3f mEigenInvK;
    bool mbInitObjectMap = false;
    vector<Object_Map*> mvNewOrChangedObj;
    float tTest[101][4] = {0};

    // frames - can be keyframes or otherwise
    Frame* mpLastFrame;
    
    // available classes
    vector<int> mvAvailableClasses;

    // object configs and priors
    unordered_map<int, ObjectConfig> mmObjectConfigs;
    unordered_map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> mmObjectModels;
    unordered_map<int, float*> mmObjectDensities;
    ObjectConfig mDefaultObjectConfig;

    //parameter
    bool mbExtendBox;
    bool mbCheckBoxEdge;
    set<int> mvIgnoreCategory;
    int mnBoxMapPoints;
    int mnMinimumContinueObs;
    float AddMPsDistMultiple;
    int mnFramesPassed = 0;

    // dataset string
    string mStrDataset;

    // line detector
    line_lbd_detect* mpLineDetect;

    //mutex
    std::mutex mMutex;
    std::mutex mMutexMapPoints;
    std::mutex mMutexNewMapPoints;


    // methods
    // separates the point cloud into instances - also filters out points that are too far away
    void ClassPointcloudsFromDepth(const cv::Mat& depth, 
                                   const cv::Mat& imgInstance, 
                                   const vector<Object_Frame>& objectFrames, 
                                   vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& vClassClouds,
                                   const float maxDepth = 3.5f);
    
    // Initialize the objects for the first time
    bool InitObjectMap(Frame* pFrame);

};

}


#endif