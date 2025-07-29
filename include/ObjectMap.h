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

#ifndef OBJECT_MAP_H
#define OBJECT_MAP_H

#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/random_sample.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include "third-party/pcl_custom/WarpPointRigid4D.h"

#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "ObjectFrame.h"
#include "Utils.h"
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

struct Cuboid
{
    // 11 DOF
    // pose
    SE3Quat mTobjw;
    // size
    double a1,a2,a3;
    float mfMaxDist;

};

struct ObjectConfig
{
    int classId;
    bool isKnown;

    struct align {
        bool isDensityBased;
        bool polyFit;
        float metricResolution;
        float normalizedResolution;
        int numSampleAngles;
    } align;

    struct icp {
        bool enabled;
        float maxIterations;
        float maxTransPercent;
        float minRotCos;
    } icp;

    struct symmetry {
        bool isRotational;
        bool isReflectional;
    } symmetry;

    struct bbox {
        float expand;
        float incrementX;
        float incrementY;
        float incrementZ;
    } bbox;

    struct pointcloud {
        size_t maxPoints;
        size_t minPoints;
    } pointcloud;

    struct downsample {
        float voxelSize;
        int minPointsPerVoxel;
    } downsample;

    struct outlierRemoval {
        bool enabled;
        size_t minNeighbors;
        float stdDev;
    } outlierRemoval;

    struct pointcloudEIF {
        bool enabled;
        float threshold;
    } pointcloudEIF;

    struct centroidTTest {
        bool enabled;
        size_t minHistorySize;
    } centroidTTest;

    struct rankSumTest {
        bool enabled;
        size_t minHistorySize;
    } rankSumTest;

    struct clustering {
        bool enabled;
        float tolerance;
    } clustering;
};

class Object_Map 
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Object_Map(Map* pMap, ObjectConfig& config, const float* priorDensity, const pcl::PointCloud<pcl::PointXYZ>::Ptr& priorCloud, float tTest[][4]);

    //isbad
    bool IsBad();

    //Set Bad
    void SetBad(const string reason);

    // Update Sum Points Pos using new add mappoints;
    void UpdateMapPoints();

    //include new accociate MPs, replaced MPs, object merge MPs
    void AddNewMapPoints(MapPoint* pMP);

    // Calculate Mean And Standard
    void CalculateMeanAndStandard();

    // Calculate position Mean And Standard, for marge objects
    void CalculatePosMeanAndStandard();

    // Filter Outlier using Reprojection
    void FilterOutlier(const Frame& CurrentFrame);
    
    // Filter Outlier using EIF
    void EIFFilterOutlier();
    
    // Calculate Object Pose by MapPoints
    void CalculateObjectPose(const Frame& CurrentFrame);

    // Calculate Yaw Error
    float CalculateYawError(const cv::Mat& SampleRwobj,const cv::Mat& twobj, const Frame& CurrentFrame,vector<vector<int>>& AssLines);

    //Calculate Object size and Shape
    void CalculateObjectShape(const bool removeOutliers = false);
    
    //updata covisibility relationship
    void UpdateCovRelation(const vector<Object_Map*>& CovObjs);

    //After associating the new MapPoints, whether the bbox projected into the image change greatly
    bool whetherAssociation(const Object_Frame& ObjFrame, const Frame& CurrentFrame);

    //Construct Bbox by reprojecting MapPoints, for data association
    void ConstructBboxByMapPoints(const Frame& CurrentFrame);

    // Merge possible same Object
    void MergeObject(Object_Map* pObj,const double CurKeyFrameStamp);

    // Align objects to their canonical pose based on the class
    void AlignToCanonical();
    int ComputeOccupancyScoreOctree(const pcl::octree::OctreePointCloud<pcl::PointXYZ>& octree, 
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const;
    int ComputeDensityScore(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const;
    pair<Eigen::Vector3d, Eigen::Vector3d> GetSizeTransFromTransform(const vector<Eigen::Vector3d>& points, const Eigen::Matrix4d& T);
    void EIFFilterOutlierCloud();

    //Get Replace Object pointer
    Object_Map* GetReplaced();

    //Get Overlap Objects by MapPoints
    vector<Eigen::Vector3f> GetCloudPoints() const;

    void InsertHistoryBboxAndTwc(const Frame& CurrentFrame);

    void AddToCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& Twc);
    void AddCloudCentroidToHistory(const Eigen::Vector3d& centroid);
    bool CentroidTTest(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const bool addToHistory = false);

    static long unsigned int nNextId;

    long unsigned int mnId; 
    static bool mnCheckMPsObs;
    static float mfEIFthreshold;
    static float MergeMPsDistMultiple;
    static int mnEIFObsNumbers;
    static bool MergeDifferentClass;
    static bool mbMonocular;
    
    long unsigned int mnCreatFrameId;
    //latest observation frame
    long unsigned int mnlatestObsFrameId;

    //bad
    bool mbBad; 
    Object_Map* mpReplaced;

    bool mbFirstInit;
    //class 
    int mnClass;
    //Observation times
    int mnObs;
    //using Tracking data association
    Bbox mLastBbox;
    Bbox mLastLastBbox;

    std::map<double,Bbox> mHistoryBbox;
    std::map<double,Eigen::Matrix4f> mHistoryTwc;
    std::map<double,Bbox> mKeyFrameHistoryBbox;
    std::map<double,Bbox> mKeyFrameHistoryBbox_Temp;

    cv::Rect mMPsProjectRect;
    
    //Used to obtain pose information
    Eigen::MatrixXd mlatestFrameLines;
    //using for project axis
    float mfLength;

    std::vector<MapPoint*> mvpMapPoints;
    std::vector<MapPoint*> mvpNewAddMapPoints;
    cv::Mat mSumPointsPos;
    //position mean and standard deviation
    vector<cv::Mat> mvHistoryPos;
    cv::Mat mHistoryPosMean;
    float mfPosStandardX,mfPosStandardY,mfPosStandardZ;
    //size
    cv::Mat mPosMean;
    float mfStandardX,mfStandardY,mfStandardZ;
    
    //Backend merge object-----------
    // potential associated objects.
    std::map<Object_Map*, int> mPossibleSameObj;     
    // object id and times simultaneous appearances .
    std::map<Object_Map*, int> mmAppearSameTimes;

    //3D BoundBox
    Cuboid mShape;

    //rotation
    double mdYaw;
    // there Type: 18 intervals  <times Score Yaw>    
    std::map<int, Eigen::Vector3d> mmYawAndScore;

    //The t stored in this T has not undergone rotation transformation
    SE3Quat mTobjw;

    // t-test statistics
    float mTTest[101][4];

    Map* mpMap;
    
    // dense object cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr mCloud;
    Eigen::Vector3d mCloudCentroidHistoryMean = Eigen::Vector3d::Zero();
    Eigen::Vector3d mCloudCentroidHistoryStdDev = Eigen::Vector3d::Zero();
    vector<Eigen::Vector3d> mvCloudCentroidHistory;
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> mvCloudClustersToCheck;

    // frozen cloud - aligned to canonical pose
    pcl::PointCloud<pcl::PointXYZ>::Ptr mFrozenCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4d mFrozenTow;

    // hyperparameters for objects
    ObjectConfig mConfig;

    // prior for alignment
    const float* mPriorDensity;
    const pcl::PointCloud<pcl::PointXYZ>::Ptr mPriorModel;

    //NeRF
    bool haveNeRF = false;
    size_t pNeRFIdx;
    Eigen::Matrix4f mTow_NeRF;
    Eigen::Vector3f BBox_NeRF;
    
    Eigen::Vector2f twc_xy;
    Eigen::Vector2f twc_xy_last = Eigen::Vector2f::Zero();
    
protected:
    //mutex
    std::mutex mMutex;
    mutable std::mutex mMutexCloud;
    mutable std::mutex mMutexMapPoints;
    std::mutex mMutexNewMapPoints;
};

}


#endif