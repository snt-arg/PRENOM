/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/18/2022
* Author: Xiao Han
*/
#ifndef SEMANTICSMANAGER_H
#define SEMANTICSMANAGER_H

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
#include "dependencies/g2o/g2o/types/se3quat.h"
#include <vector>
#include <map>
#include <mutex>

using namespace g2o;
using namespace std;

namespace ORB_SLAM2
{
class Frame;
class MapPoint;
class Map;
class Object_Map;

class SemanticsManager
{
public:

    struct Task
    {
        Frame* frame;
        cv::Mat imgGray;
        cv::Mat imgInstance;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    };


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    SemanticsManager(Map* pMap, const string &strDataset, const string &strSettingPath);
    void AddTaskToQueue(Task task);
    void Run();

    
protected:

    // Map
    Map* mpMap;

    //dataset path
    string mstrDataset;

    // frame queue
    std::queue<Task> mTaskQueue;

    // parameters from Tracking thread
    int mnImgWidth;
    int mnImgHeight;
    cv::Mat mDistCoef;
    cv::Mat mK;

    bool mbInitObjectMap;
    vector<Object_Map*> mvNewOrChangedObj;
    //t test 
    float tTest[101][4] = {0};

    //parameter
    bool mbExtendBox;
    bool mbCheckBoxEdge;
    set<int> mvIgnoreCategory;
    int mnBoxMapPoints;
    int mnMinimumContinueObs;
    float AddMPsDistMultiple;

    // dataset string
    string mStrDataset;

    // line detector
    line_lbd_detect* mpLineDetect;

    //mutex
    std::mutex mMutex;
    std::mutex mMutexMapPoints;
    std::mutex mMutexNewMapPoints;

};

}


#endif