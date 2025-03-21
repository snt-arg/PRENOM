/**
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/18/2022
* Author: Xiao Han
*
* Modification: PRENOM
* Version: 1.0
* Created: 12/25/2024
* Author: Saad Ejaz
*/

#include "ObjectMap.h"
#include "Map.h"
#include "Converter.h"
#include "OptimizeObject.h"
#include <chrono>
#include "EIF.h"
#include <omp.h>

#define GRID_SIZE 64
#define GRID_SIZE_SQ 4096

namespace ORB_SLAM2
{
long unsigned int Object_Map::nNextId=0;
bool Object_Map::mnCheckMPsObs = false;
float Object_Map::mfEIFthreshold, Object_Map::MergeMPsDistMultiple;
int Object_Map::mnEIFObsNumbers;
bool Object_Map::MergeDifferentClass = false;
bool Object_Map::mbMonocular = false;


Object_Map::Object_Map(Map* pMap, ObjectConfig& objConfig, const float* priorDensity, const pcl::PointCloud<pcl::PointXYZ>::Ptr& priorCloud, float tTest[][4]) : 
    mbBad(false),mbFirstInit(true),mnObs(0),mpMap(pMap), mSumPointsPos(cv::Mat::zeros(3,1,CV_32F)), mTobjw(SE3Quat()),
    mpReplaced(static_cast<Object_Map*>(NULL)), mConfig(objConfig), mPriorDensity(priorDensity), mPriorModel(priorCloud)
{
    mnId=nNextId++;
    mCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    // the t-test data
    for(int i=0;i<101;i++)
    {
        for(int j=0;j<4;j++)
            mTTest[i][j] = tTest[i][j];
    }
}

bool Object_Map::IsBad()
{
    unique_lock<mutex> lock(mMutex);
    return mbBad;  
}

void Object_Map::SetBad(const string reason)
{
    unique_lock<mutex> lock(mMutex);
    unique_lock<mutex> lock1(mMutexMapPoints);
    unique_lock<mutex> lock2(mMutexNewMapPoints);

    for(MapPoint* pMP : mvpMapPoints)
    {
        pMP->EraseObject(this);
    }

    //cout<<"mnId: "<<mnId<<" Class: "<<mnClass<<" reason: "<<reason<<endl;
    mbBad = true;
}


//include new accociate MPs, replaced MPs, object merge MPs
void Object_Map::AddNewMapPoints(MapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexNewMapPoints);
    mvpNewAddMapPoints.push_back(pMP);
    
}

// Update MapPoints using new add mappoints;
void Object_Map::UpdateMapPoints()
{
    if(mvpNewAddMapPoints.empty())
        return;
        
    unique_lock<mutex> lock(mMutexMapPoints);

    set<MapPoint*> mvpMPs(mvpMapPoints.begin(),mvpMapPoints.end());

    for(MapPoint* pMP : mvpNewAddMapPoints)
    {   
        if(mvpMPs.find(pMP) != mvpMPs.end())
            continue;
        pMP->AddObject(this);
        mvpMapPoints.push_back(pMP);

    }
    mvpNewAddMapPoints.clear();

    // std::cout << "Number of MapPoints in Object: " << mvpMapPoints.size() << std::endl;
}

//Calculate the mean and standard deviation
void Object_Map::CalculateMeanAndStandard()
{
    if(IsBad())
        return;
    
    unique_lock<mutex> lock(mMutexMapPoints);

    mSumPointsPos = cv::Mat::zeros(3,1,CV_32F);
    for(MapPoint* pMP : mvpMapPoints)
    {   
        mSumPointsPos += pMP->GetWorldPos();
    }
    mPosMean = mSumPointsPos / mvpMapPoints.size();
    
}

void Object_Map::EIFFilterOutlier()
{   

    unique_lock<mutex> lock(mMutexMapPoints);

    //Extended Isolation Forest
    std::mt19937 rng(12345);
	std::vector<std::array<float, 3>> data;
    
    if(mKeyFrameHistoryBbox.size() < 5 || mvpMapPoints.size() < 20)
        return;
    
	for (size_t i = 0; i < mvpMapPoints.size(); i++)
	{
		std::array<float, 3> temp;
        cv::Mat pos = mvpMapPoints[i]->GetWorldPos();
		for (uint32_t j = 0; j < 3; j++)
		{   
			temp[j] = pos.at<float>(j);
		}
		data.push_back(temp);
	}
    
    //auto t1 = std::chrono::system_clock::now();

	EIF::EIForest<float, 3> forest;
    
    double th = mfEIFthreshold;
    
    //Appropriately expand the EIF threshold for non-textured objects
    if(mnClass == 73 || mnClass == 46 || mnClass == 41 || mnClass == 62)
    {
        th = th + 0.02;
    }

    double th_serious = th + 0.1;

    int point_num = 0;
    if(mvpMapPoints.size() > 100)
        point_num = mvpMapPoints.size() / 2;
    else
        point_num = mvpMapPoints.size() * 2 / 3;

	if(!forest.Build(40, 12345, data,point_num))
	{
		std::cerr << "Failed to build Isolation Forest.\n";
	}
    
	std::vector<double> anomaly_scores;

	if(!forest.GetAnomalyScores(data, anomaly_scores))
	{
		std::cerr << "Failed to calculate anomaly scores.\n";
	}
    
    vector<MapPoint*> newVpMapPoints;
    for(size_t i = 0,iend = mvpMapPoints.size();i<iend;i++)
    {   
        MapPoint* pMP = mvpMapPoints[i];

        //outlier                   If the point is added for a long time, it is considered stable
        if(mnCheckMPsObs)
        {
            if(anomaly_scores[i] > th_serious)
            {
                pMP->EraseObject(this);
            }
            else if(anomaly_scores[i] > th && mnlatestObsFrameId - pMP->mAssociateObjects[this] < mnEIFObsNumbers)
            {
                pMP->EraseObject(this);
            }
            else
                newVpMapPoints.push_back(pMP);
        }
        else
        {
            if(anomaly_scores[i] > th)
            {
                pMP->EraseObject(this);
            }
            else
                newVpMapPoints.push_back(pMP);
        } 

    }

    mvpMapPoints = newVpMapPoints;
    //auto t2 = std::chrono::system_clock::now();
    //std::cout<< "EIF time: "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()<<std::endl;

}

void Object_Map::EIFFilterOutlierCloud()
{   

    // lock before the function
    // unique_lock<mutex> lock(mMutexCloud);

    //Extended Isolation Forest
    std::mt19937 rng(12345);
	std::vector<std::array<float, 3>> data;
    
    const size_t cloudSize = mCloud->points.size();

    if(mKeyFrameHistoryBbox.size() < 5 || cloudSize < mConfig.pointcloud.maxPoints/3)
        return;
    
    for (size_t i = 0; i < cloudSize; i++)
    {
        std::array<float, 3> temp;
        temp[0] = mCloud->points[i].x;
        temp[1] = mCloud->points[i].y;
        temp[2] = mCloud->points[i].z;
        data.push_back(temp);
    }

	EIF::EIForest<float, 3> forest;
	if(!forest.Build(40, 12345, data, cloudSize / 2))
		std::cerr << "Failed to build Isolation Forest.\n";
    
	std::vector<double> anomaly_scores;
	if(!forest.GetAnomalyScores(data, anomaly_scores))
		std::cerr << "Failed to calculate anomaly scores.\n";
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(size_t i = 0,iend = cloudSize;i<iend;i++)
        if(anomaly_scores[i] < mConfig.pointcloudEIF.threshold)
            newCloud->push_back(mCloud->points[i]);
    mCloud = newCloud;
}


void Object_Map::FilterOutlier(const Frame& CurrentFrame)
{

    unique_lock<mutex> lock(mMutexMapPoints);

    bool Reprojection = true;

    if(mnlatestObsFrameId != CurrentFrame.mnId)
        Reprojection = false;
    
    //Make sure the Bbox is not at the edge of the image
    if(mLastBbox.x < CurrentFrame.mnMinX + 30 || mLastBbox.x + mLastBbox.width > CurrentFrame.mnMaxX - 30)
        Reprojection = false;
    if(mLastBbox.y < CurrentFrame.mnMinY + 30 || mLastBbox.y + mLastBbox.height > CurrentFrame.mnMaxY - 30)
        Reprojection = false;
    
    //Too small Bbox means a long distance and is prone to errors
    if(mLastBbox.area() < (CurrentFrame.mnMaxX -CurrentFrame.mnMinX) * (CurrentFrame.mnMaxY - CurrentFrame.mnMinY) * 0.05)
        Reprojection = false;
    
    //Reprojection Filter Outlier
    //now it is CurrentFrame Bbox
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    for(vector<MapPoint*>::iterator it=mvpMapPoints.begin();it!=mvpMapPoints.end();)
    {
        
        if((*it)->isBad())
        {   

            (*it)->EraseObject(this);
            (*it) = mvpMapPoints.back();
            mvpMapPoints.pop_back();
            continue;
        }
        
        if(Reprojection)
        {
            cv::Mat FramePos = Rcw * (*it)->GetWorldPos() + tcw;
            float invz = 1.0 / FramePos.at<float>(2);
            // camera -> image.
            float u = CurrentFrame.fx * FramePos.at<float>(0) * invz + CurrentFrame.cx;
            float v = CurrentFrame.fy * FramePos.at<float>(1) * invz + CurrentFrame.cy;
            cv::Point point(u,v);
            if(!mLastBbox.contains(point))
            {
                mSumPointsPos -= (*it)->GetWorldPos();
                (*it)->EraseObject(this);
                (*it) = mvpMapPoints.back();
                mvpMapPoints.pop_back();
            }
            else
                ++it;           
        }
        else
            ++it;
    } 
}

//Calculate the mean and standard deviation
void Object_Map::CalculatePosMeanAndStandard()
{
    if(mbBad)
        return;

    unique_lock<mutex> lock(mMutex);

    cv::Mat mSumHistoryPos = cv::Mat::zeros(3,1,CV_32F);
    for(const cv::Mat& Pos : mvHistoryPos)
        mSumHistoryPos += Pos;
    mHistoryPosMean = mSumHistoryPos / mvHistoryPos.size();

    float meanX = mHistoryPosMean.at<float>(0);
    float meanY = mHistoryPosMean.at<float>(1);
    float meanZ = mHistoryPosMean.at<float>(2);

    float sumX = 0, sumY = 0, sumZ = 0;
    for(const cv::Mat& Pos : mvHistoryPos)
    {
        sumX += (meanX - Pos.at<float>(0)) * (meanX - Pos.at<float>(0));
        sumY += (meanY - Pos.at<float>(1)) * (meanY - Pos.at<float>(1));
        sumZ += (meanZ - Pos.at<float>(2)) * (meanZ - Pos.at<float>(2));
    }
    mfPosStandardX = sqrt(sumX / mvHistoryPos.size());
    mfPosStandardY = sqrt(sumY / mvHistoryPos.size());
    mfPosStandardZ = sqrt(sumZ / mvHistoryPos.size());

}

void Object_Map::CalculateObjectPose(const Frame& CurrentFrame)
{
    if(mbBad)
        return;
    
    //Note that there are two translations here, 
    //  Tobjw   mshape->Tobjw
    //because the same group of map points have different center points in the world coordinate system
    //and the object coordinate system under the action of object rotation
    //Since the rotation is not considered in the translation of the Object_Frame, 
    //in order to maintain consistency when associating the Object_Frame and the Object_Map, an additional t is stored here.
    
    cv::Mat twobj = cv::Mat::zeros(3,1,CV_32F);
    Eigen::VectorXd robustX, robustY, robustZ;
    size_t dimSize;
    
    if (mbMonocular) {
        // only map points available
        unique_lock<mutex> lock(mMutexMapPoints);
        vector<double> vectorX, vectorY, vectorZ;
        for (const auto& pMP : mvpMapPoints)
        {
            if (pMP->isBad())
                continue;

            cv::Mat Pos = pMP->GetWorldPos();
            Eigen::Vector3d Pos_eigen = Converter::toVector3d(Pos);
            vectorX.push_back(Pos_eigen(0));
            vectorY.push_back(Pos_eigen(1));
            vectorZ.push_back(Pos_eigen(2));
        }

        dimSize = vectorX.size();
        robustX = Eigen::VectorXd::Zero(dimSize);
        robustY = Eigen::VectorXd::Zero(dimSize);
        robustZ = Eigen::VectorXd::Zero(dimSize);

        for (size_t i = 0; i < dimSize; i++)
        {
            robustX(i) = vectorX[i];
            robustY(i) = vectorY[i];
            robustZ(i) = vectorZ[i];
        }
    }
    else {
        // use the dense point cloud
        unique_lock<mutex> lock(mMutexCloud);
        if (mCloud->empty())
            return;

        dimSize = mCloud->size();
        robustX = Eigen::VectorXd::Zero(dimSize);
        robustY = Eigen::VectorXd::Zero(dimSize);
        robustZ = Eigen::VectorXd::Zero(dimSize);
        for (size_t i = 0; i < dimSize; i++)
        {
            robustX(i) = mCloud->points[i].x;
            robustY(i) = mCloud->points[i].y;
            robustZ(i) = mCloud->points[i].z;
        }
    }
    
    // sort in all axes
    sort(robustX.begin(), robustX.end());
    sort(robustY.begin(), robustY.end());
    sort(robustZ.begin(), robustZ.end());

    // get the size - assume it's the same for all axes
    assert(robustY.size() == dimSize && robustZ.size() == dimSize);

    twobj.at<float>(0) = (robustX(0) + robustX(dimSize-1)) / 2;
    twobj.at<float>(1) = (robustY(0) + robustY(dimSize-1)) / 2;
    twobj.at<float>(2) = (robustZ(0) + robustZ(dimSize-1)) / 2;
    
    //using for project axis, to calculate rotation
    vector<float> length;
    length.push_back((robustX(dimSize-1) - robustX(0)) / 2);
    length.push_back((robustY(dimSize-1) - robustY(0)) / 2);
    length.push_back((robustZ(dimSize-1) - robustZ(0)) / 2);
    sort(length.begin(),length.end());
    mfLength = length[2];
    
    //calculate and update yaw
    if(mlatestFrameLines.rows() > 2 && !mLastBbox.mbEdgeAndSmall)
    {
        cv::Mat SampleRwobj;
        //calculate object Rotation

        vector<vector<int>> AssLines;
        vector<vector<int>> BestAssLines;
        Eigen::Matrix3d yawR;

        // -45° - 45°    90° / 5° = 18;
        //Take the middle
        // -42.5° - 42.5°    85° / 5° = 17;

        float sampleYaw = 0;
        float bestYaw = 0;
        float bestScore = 0;
        int bestIdx = -1;

        for(int i=0;i<18;i++)
        {   
            //sample yaw
            sampleYaw = (i * 5.0 - 42.5) / 180.0 * M_PI;
            yawR = Converter::eulerAnglesToMatrix(sampleYaw);
            SampleRwobj = Converter::toCvMat(yawR);
            
            AssLines.clear();
            float score = CalculateYawError(SampleRwobj,twobj,CurrentFrame,AssLines);

            if( score > bestScore)
            {   
                // two direction have association 
                if(!AssLines[0].empty() || !AssLines[1].empty())
                {
                    bestScore = score;
                    bestYaw = sampleYaw;
                    bestIdx = i;
                    BestAssLines = AssLines;
                }   
            }
        }
        //cout << "bestScore: "<<bestScore<<endl;
        
        if(bestScore != 0)
        {   
            //Refine rotation estimation
            float OptimizeYaw = ObjectOptimizer::OptimizeRotation(*this,BestAssLines,bestYaw,twobj,CurrentFrame);
            //cout << "OptimizeYaw: "<<OptimizeYaw<<endl;
            if(abs(bestYaw - OptimizeYaw) < 0.087266)  // 5/180 * PI
                bestYaw = OptimizeYaw;
        }
        
        //update yaw (history)
        if(bestScore!=0)
        {
            if(mmYawAndScore.count(bestIdx))
            {   
                //update times, score, yaw
                Eigen::Vector3d& Item = mmYawAndScore[bestIdx];
                Item(0) += 1.0; 
                Item(1) = (Item[1] * (1 - 1/Item(0)) + bestScore * 1/Item(0));
                Item(2) = (Item[2] * (1 - 1/Item(0)) + bestYaw * 1/Item(0)); 
            }
            else
            {   
                Eigen::Vector3d Item(1.0,bestScore,bestYaw);
                mmYawAndScore[bestIdx] = Item;
            }   
        }
    }
    else if(mnObs > 50 &&  mvpMapPoints.size() > 50)
    {
        //PCA
        Eigen::MatrixXd points(2,dimSize);
        points.row(0) = robustX;
        points.row(1) = robustY;
        double meanX = points.row(0).mean();
        double meanY = points.row(1).mean();

        points.row(0) = points.row(0) - Eigen::MatrixXd::Ones(1,dimSize) * meanX;
        points.row(1) = points.row(1) - Eigen::MatrixXd::Ones(1,dimSize) * meanY;
        
        Eigen::Matrix2d covariance = points * points.transpose() / double(dimSize);
        double ratio = max(covariance(0,0),covariance(1,1)) / min(covariance(0,0),covariance(1,1));

        double score = 0;
        double yaw = 0;
        int yawIdx = 0;
        //The standard deviation is greater than 1.1
        if(ratio > 1.21)
        {
            Eigen::EigenSolver<Eigen::Matrix2d> es(covariance);
            Eigen::Matrix2d EigenVectors = es.pseudoEigenvectors();
            Eigen::Matrix2d EigenValues = es.pseudoEigenvalueMatrix();
            //cout<<"covariance: "<<covariance<<endl;
        
            yaw = atan2(EigenVectors(1,0),EigenVectors(0,0)) * 180.0 / M_PI;
            if(yaw > 45.0 && yaw < 135.0)
                yaw = yaw - 90;
            else if(yaw >= 135.0)
                yaw = yaw  - 180.0;
            else if(yaw <= -135.0)
                yaw = 180 + yaw;
            else if(yaw < -45.0 && yaw > -135.0)
                yaw = 90 + yaw;
            
            yawIdx = int(abs(yaw + 42.5 / 5.0));
            yaw = yaw / 180.0 * M_PI;

            // 0 - 1 The score of PCA is less than that of projection
            score = mvpMapPoints.size() / mnObs;
            if(score > 5)
                score = 1;
        }

        if(score!=0)
        {   
        
            if(mmYawAndScore.count(yawIdx))
            {   
                //update times, score, yaw
                Eigen::Vector3d& Item = mmYawAndScore[yawIdx];
                Item(0) += 1.0; 
                Item(1) = (Item[1] * (1 - 1/Item(0)) + score * 1/Item(0));
                Item(2) = (Item[2] * (1 - 1/Item(0)) + yaw * 1/Item(0)); 
            }
            else
            {   
                Eigen::Vector3d Item(1.0,score,yaw);
                mmYawAndScore[yawIdx] = Item;
            }   
        }

    }

    //get the result yaw
    float resYaw = 0;
    
    if(!mmYawAndScore.empty())
    {
        vector<Eigen::Vector3d> YawAndScore;
        for(std::map<int,Eigen::Vector3d>::iterator it = mmYawAndScore.begin();it!=mmYawAndScore.end();it++)
        {
            YawAndScore.push_back(it->second);
        }

        if(YawAndScore.size() > 1)
        {
            sort(YawAndScore.begin(),YawAndScore.end(),[](const Eigen::Vector3d& v1,const Eigen::Vector3d& v2){return v1(1) > v2(1);});
            if(YawAndScore[0](0) > mnObs / 4.0)
                resYaw = YawAndScore[0](2);
            else if(YawAndScore[0](0) > mnObs / 6.0 && YawAndScore[0](0) > YawAndScore[1](0))
                resYaw = YawAndScore[0](2);
            else
            {
                sort(YawAndScore.begin(),YawAndScore.end(),[](const Eigen::Vector3d& v1,const Eigen::Vector3d& v2){return v1(0) > v2(0);});
                resYaw = YawAndScore[0](2);
            }
        }
        else
        {
            resYaw = YawAndScore[0](2);
        }
    }
    
    //cout <<"resYaw: "<< resYaw <<endl;
    Eigen::Matrix3d Rwobj =Converter::eulerAnglesToMatrix(resYaw);
    mTobjw = SE3Quat(Rwobj,Converter::toVector3d(twobj));
    mTobjw = mTobjw.inverse();

}

float Object_Map::CalculateYawError(const cv::Mat& SampleRwobj,const cv::Mat& twobj, const Frame& CurrentFrame,vector<vector<int>>& AssLines)
{
    //project axix to frame
    // center  X Y Z(3 points on the axis)

    vector<cv::Mat> PointPos;
    cv::Mat center = cv::Mat::zeros(3,1,CV_32F);
    PointPos.push_back(center);
    cv::Mat center_X = cv::Mat::zeros(3,1,CV_32F);
    center_X.at<float>(0) = mfLength;
    PointPos.push_back(center_X);
    cv::Mat center_Y = cv::Mat::zeros(3,1,CV_32F);
    center_Y.at<float>(1) = mfLength;
    PointPos.push_back(center_Y);
    cv::Mat center_Z = cv::Mat::zeros(3,1,CV_32F);
    center_Z.at<float>(2) = mfLength;
    PointPos.push_back(center_Z);

    //Project 
    vector<cv::Point2f> points;
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    for(const cv::Mat& Pos : PointPos)
    {
        cv::Mat framePos =  Rcw * (SampleRwobj * Pos + twobj) + tcw;
        float inv_z = 1.0 / framePos.at<float>(2);
        float u = CurrentFrame.fx * framePos.at<float>(0) * inv_z + CurrentFrame.cx;
        float v = CurrentFrame.fy * framePos.at<float>(1) * inv_z + CurrentFrame.cy;
        points.emplace_back(u,v);

    }
    //calculate angle
    //O-X

    float angleX;
    if(points[0].x < points[1].x)
        angleX = atan2(points[1].y - points[0].y, points[1].x - points[0].x);
    else
        angleX = atan2(points[0].y - points[1].y, points[0].x - points[1].x);

    //O-Y
    float angleY;
    if(points[0].x < points[2].x)
        angleY = atan2(points[2].y - points[0].y, points[2].x - points[0].x);
    else
        angleY = atan2(points[0].y - points[2].y, points[0].x - points[2].x);

    //O-Z
    float angleZ;
    if(points[0].x < points[3].x)
        angleZ = atan2(points[3].y - points[0].y, points[3].x - points[0].x);
    else
        angleZ = atan2(points[0].y - points[3].y, points[0].x - points[3].x);


    float error = 0;
    int num = 0;
    //th = 5, Lines with an error of less than 5 degrees are considered relevant
    float th = 5;

    //associate lines, for optimizer rotation
    vector<int> AssLinesX, AssLinesY, AssLinesZ;

    for(int i=0; i < mlatestFrameLines.rows();i++)
    {
        double x1 = mlatestFrameLines(i,0);
        double y1 = mlatestFrameLines(i,1);
        double x2 = mlatestFrameLines(i,2);
        double y2 = mlatestFrameLines(i,3);

        float angle = atan2(y2 - y1, x2 - x1);

        //3 lines angle error  0 ~ Pi/2
        float angle_error_X = abs((angle - angleX) * 180.0 / M_PI);
        angle_error_X = min(angle_error_X ,180 - angle_error_X);
        float angle_error_Y = abs((angle - angleY) * 180.0 / M_PI);
        angle_error_Y = min(angle_error_Y ,180 - angle_error_Y);
        float angle_error_Z = abs((angle - angleZ) * 180.0 / M_PI);
        angle_error_Z = min(angle_error_Z ,180 - angle_error_Z);

        float minError = min(min(angle_error_X,angle_error_Y),angle_error_Z);
        //cout<<"line: "<<i<<" minError: " <<minError<<endl;

        if(minError < th)
        {
            error += minError;
            ++num;
            if(minError == angle_error_X)
                AssLinesX.push_back(i);
            else if (minError == angle_error_Y)
                AssLinesY.push_back(i);
            else
                AssLinesZ.push_back(i);
        }

    }
    
    if(num == 0)
        return 0;
    else
    {
        AssLines.push_back(AssLinesX);
        AssLines.push_back(AssLinesY);
        AssLines.push_back(AssLinesZ);

        //The more associated lines and the smaller the error, the better
        float score = (float(num) / mlatestFrameLines.rows()) * (5 - error/num);
        return score;
    }

}

// Calculate size
void Object_Map::CalculateObjectShape(const bool removeOutliers)
{
    if(mbBad)
        return;

    cv::Mat tobjw_shape = cv::Mat::zeros(3,1,CV_32F);

    Eigen::Matrix3d R = mTobjw.to_homogeneous_matrix().block(0,0,3,3);
    cv::Mat Robjw = Converter::toCvMat(R);
    Eigen::Matrix4d Tow = Eigen::Matrix4d::Identity();
    Tow.block(0,0,3,3) = R;

    Eigen::VectorXd robustX, robustY, robustZ;
    pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud(new pcl::PointCloud<pcl::PointXYZ>);
    size_t dimSize;
    if (mbMonocular)
    {
        unique_lock<mutex> lock(mMutexMapPoints);
        vector<double> vectorX, vectorY, vectorZ;

        for(const auto& pMP : mvpMapPoints)
        {   
            if(pMP->isBad())
                    continue;
            
            cv::Mat Pos = pMP->GetWorldPos();
            Eigen::Vector3d ObjPos = R * Converter::toVector3d(Pos);
            vectorX.push_back(ObjPos(0));
            vectorY.push_back(ObjPos(1));
            vectorZ.push_back(ObjPos(2));
        }

        dimSize = vectorX.size();
        robustX = Eigen::VectorXd::Zero(dimSize);
        robustY = Eigen::VectorXd::Zero(dimSize);
        robustZ = Eigen::VectorXd::Zero(dimSize);

        for (size_t i = 0; i < dimSize; i++)
        {
            robustX(i) = vectorX[i];
            robustY(i) = vectorY[i];
            robustZ(i) = vectorZ[i];
        }
    }
    else {
        // get pointcloud in object coordinate
        unique_lock<mutex> lock(mMutexCloud);

        if (mCloud->empty())
            return;

        // remove outliers and transform to object coordinate
        if (mConfig.pointcloudEIF.enabled)
            EIFFilterOutlierCloud();
        pcl::transformPointCloud(*mCloud,*objCloud,Tow);

        dimSize = objCloud->size();
        robustX = Eigen::VectorXd::Zero(dimSize);
        robustY = Eigen::VectorXd::Zero(dimSize);
        robustZ = Eigen::VectorXd::Zero(dimSize);

        for (size_t i = 0; i < dimSize; i++)
        {
            robustX(i) = objCloud->points[i].x;
            robustY(i) = objCloud->points[i].y;
            robustZ(i) = objCloud->points[i].z;
        }
    }
    
    // sort in all axes    
    sort(robustX.begin(), robustX.end());
    sort(robustY.begin(), robustY.end());
    sort(robustZ.begin(), robustZ.end());
    
    // get the size - assume it's the same for all axes
    assert(robustY.size() == dimSize && robustZ.size() == dimSize);

    //pos after Robjw
    tobjw_shape.at<float>(0) = -(robustX(0) + robustX(dimSize-1)) / 2;
    tobjw_shape.at<float>(1) = -(robustY(0) + robustY(dimSize-1)) / 2;
    tobjw_shape.at<float>(2) = -(robustZ(0) + robustZ(dimSize-1)) / 2;

    if(mbFirstInit)
    {
        Cuboid shape;
        shape.mTobjw = mTobjw;
        mShape = shape;
        mbFirstInit = false;
    }

    if(haveNeRF)
        return;

    mShape.mTobjw = SE3Quat(R,Converter::toVector3d(tobjw_shape));
    mShape.a1 = abs(robustX(dimSize-1) - robustX(0)) / 2;
    mShape.a2 = abs(robustY(dimSize-1) - robustY(0)) / 2;
    mShape.a3 = abs(robustZ(dimSize-1) - robustZ(0)) / 2;
    mShape.mfMaxDist = sqrt(mShape.a1 * mShape.a1 + mShape.a2 * mShape.a2 + mShape.a3 * mShape.a3);

    if (!mbMonocular && removeOutliers)
    {
        // remove points outside the bounding box
        pcl::PointCloud<pcl::PointXYZ>::Ptr objCloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
        vector<double> X_axis,Y_axis,Z_axis;
        for (const pcl::PointXYZ& point : objCloud->points)
        {
            if (point.x >= robustX(0) && point.x <= robustX(dimSize-1) &&
                point.y >= robustY(0) && point.y <= robustY(dimSize-1) &&
                point.z >= robustZ(0) && point.z <= robustZ(dimSize-1))
            {
                objCloudFiltered->points.push_back(point);
                X_axis.push_back(point.x);
                Y_axis.push_back(point.y);
                Z_axis.push_back(point.z);
            }
        }
        // transform back to world coordinate
        pcl::transformPointCloud(*objCloudFiltered,*mCloud,Tow.inverse().cast<float>());

        // copy to the frozen cloud
        sort(X_axis.begin(),X_axis.end());
        sort(Y_axis.begin(),Y_axis.end());
        sort(Z_axis.begin(),Z_axis.end());

        // new translation
        cv::Mat newTobjw = cv::Mat::zeros(3,1,CV_32F);
        newTobjw.at<float>(0) = -(X_axis[0] + X_axis[X_axis.size()-1]) / 2;
        newTobjw.at<float>(1) = -(Y_axis[0] + Y_axis[Y_axis.size()-1]) / 2;
        newTobjw.at<float>(2) = -(Z_axis[0] + Z_axis[Z_axis.size()-1]) / 2;
        Tow.block(0,3,3,1) = Converter::toVector3d(newTobjw);

        pcl::copyPointCloud(*mCloud,*mFrozenCloud);
        mFrozenTow = Tow;
    }
}

//step5. updata covisibility relationship
void Object_Map::UpdateCovRelation(const vector<Object_Map*>& CovObjs)
{
    if(mbBad)
        return;

    unique_lock<mutex> lock(mMutex);
    for(Object_Map* pObj : CovObjs)
    {
        if(pObj == this)
            continue;
        if(pObj->IsBad())
            continue;

        mmAppearSameTimes[pObj]++;
    }

}

//After associating the new MapPoints, whether the bbox projected into the image change greatly
bool Object_Map::whetherAssociation(const Object_Frame& ObjFrame,const Frame& CurrentFrame)
{
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    float fx = CurrentFrame.fx;
    float fy = CurrentFrame.fy;
    float cx = CurrentFrame.cx;
    float cy = CurrentFrame.cy;

    
    vector<float> xpt,ypt,mix_xpt,mix_ypt;

    // original
    for(MapPoint* pMP : mvpMapPoints)
    {
        if(pMP->isBad())
            continue;

        cv::Mat pos = pMP->GetWorldPos();
        pos = Rcw * pos + tcw;
        float inv_z = 1.0 / pos.at<float>(2);
        float u =  fx * pos.at<float>(0) * inv_z + cx;
        float v =  fy * pos.at<float>(1) * inv_z + cy;

        xpt.push_back(u);
        mix_xpt.push_back(u);
        ypt.push_back(v);
        mix_ypt.push_back(v);

    }

    //mix
    for(MapPoint* pMP : ObjFrame.mvpMapPoints)
    {
        if(pMP->isBad())
            continue;

        cv::Mat pos = pMP->GetWorldPos();
        pos = Rcw * pos + tcw;
        float inv_z = 1.0 / pos.at<float>(2);
        float u =  fx * pos.at<float>(0) * inv_z + cx;
        float v =  fy * pos.at<float>(1) * inv_z + cy;

        mix_xpt.push_back(u);
        mix_ypt.push_back(v);
    }
    
    sort(xpt.begin(),xpt.end());
    sort(mix_xpt.begin(),mix_xpt.end());
    sort(ypt.begin(),ypt.end());
    sort(mix_ypt.begin(),mix_ypt.end());
    
    cv::Rect origin(xpt[0],ypt[0],xpt[xpt.size()-1] - xpt[0],ypt[ypt.size()-1] - ypt[0]);
    cv::Rect mix(mix_xpt[0],mix_ypt[0],mix_xpt[mix_xpt.size()-1] - mix_xpt[0],mix_ypt[mix_ypt.size()-1] - mix_ypt[0]);
    
    float IoUarea = (origin & mix).area();
    IoUarea = IoUarea / (origin.area() + mix.area() - IoUarea);
    if(IoUarea < 0.4 )
        return false;
    else
        return true;

}

//Construct Bbox by reprojecting MapPoints, for data association
void Object_Map::ConstructBboxByMapPoints(const Frame& CurrentFrame)
{
    if(mbBad)
        return;
    unique_lock<mutex> lock(mMutexMapPoints);

    vector<float> v_u;
    vector<float> v_v;
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    float fx = CurrentFrame.fx;
    float fy = CurrentFrame.fy;
    float cx = CurrentFrame.cx;
    float cy = CurrentFrame.cy;
    for(MapPoint* pMP : mvpMapPoints)
    {   
        // world -> camera.
        cv::Mat FramePos = Rcw * pMP->GetWorldPos() + tcw;
        float invz = 1.0 / FramePos.at<float>(2);
        // camera -> image.
        float u = fx * FramePos.at<float>(0) * invz + cx;
        float v = fy * FramePos.at<float>(1) * invz + cy;
        v_u.push_back(u);
        v_v.push_back(v);
    }

    sort(v_u.begin(),v_u.end());
    sort(v_v.begin(),v_v.end());

    // make insure in the image
    float minU = max(CurrentFrame.mnMinX,v_u[0]);
    float minV = max(CurrentFrame.mnMinY,v_v[0]);
    float maxU = min(CurrentFrame.mnMaxX,v_u[v_u.size()-1]);
    float maxV = min(CurrentFrame.mnMaxY,v_v[v_v.size()-1]);

    mMPsProjectRect = cv::Rect(minU,minV,maxU-minU,maxV-minV);

}

void Object_Map::MergeObject(Object_Map* pObj,const double CurKeyFrameStamp)
{
    //cout << "MergeObject: "<<pObj->mnClass<<endl;
    if(pObj->IsBad())
        return;
    
    unique_lock<mutex> lock(mMutex);

    //update
    if(pObj->mnCreatFrameId < mnCreatFrameId)
        mnCreatFrameId = pObj->mnCreatFrameId;
    if(pObj->mnlatestObsFrameId > mnlatestObsFrameId)
    {
        mnlatestObsFrameId = pObj->mnlatestObsFrameId; 
        mLastBbox = pObj->mLastBbox;
        mLastLastBbox = pObj->mLastBbox;
        mlatestFrameLines = pObj->mlatestFrameLines; 
    }
    mnObs += pObj->mnObs;

    bool checkMPs = false;
    SE3Quat Tobjw;
    float Maxdist_x = 0;
    float Maxdist_y = 0;
    float Maxdist_z = 0;
    if(mvpMapPoints.size() > 10)
    {
        checkMPs = true;
        if(mbFirstInit)
        {
            Tobjw = mTobjw;
            Maxdist_x = mfLength;
            Maxdist_y = mfLength;
            Maxdist_z = mfLength;
        }
        else
        {   //more accurate
            Tobjw = mShape.mTobjw;
            Maxdist_x = mShape.a1;
            Maxdist_y = mShape.a2;
            Maxdist_z = mShape.a3;
        }
    }

    for(size_t j=0;j<pObj->mvpMapPoints.size();j++)
    {   
        MapPoint* pMP = pObj->mvpMapPoints[j];
        if(pMP->isBad())
            continue;

        // check position
        if(checkMPs)
        {
            Eigen::Vector3d ObjPos = Tobjw * Converter::toVector3d(pMP->GetWorldPos());
            if(abs(ObjPos(0)) > MergeMPsDistMultiple * Maxdist_x || abs(ObjPos(1)) > MergeMPsDistMultiple * Maxdist_y || abs(ObjPos(2)) > MergeMPsDistMultiple * Maxdist_z)
                continue;
        }
        
        //new MapPoint
        AddNewMapPoints(pMP);
    }
    UpdateMapPoints();   
    
    //Fiter outlier
    EIFFilterOutlier();

    //update history pos
    for(const cv::Mat& pos : pObj->mvHistoryPos)
        mvHistoryPos.push_back(pos);

    //update covisibility relationship
    map<Object_Map*,int>::iterator it;
    for(it = pObj->mmAppearSameTimes.begin();it!= pObj->mmAppearSameTimes.end();it++)
    {
        mmAppearSameTimes[it->first] += it->second;
    }

    //update nerf bbox
    for(const auto& it : pObj->mHistoryBbox)
    {   
        double stamp = it.first;
        if(mHistoryBbox.find(stamp) != mHistoryBbox.end())
        {
            mHistoryBbox[stamp] = it.second;
            mHistoryTwc[stamp] = pObj->mHistoryTwc[stamp];
            if(CurKeyFrameStamp == stamp)
            {
                mKeyFrameHistoryBbox[stamp] = it.second;
                mKeyFrameHistoryBbox_Temp[stamp] = it.second;
            }
        }
            
    }

}

Object_Map* Object_Map::GetReplaced()
{
    unique_lock<mutex> lock(mMutex);
    return mpReplaced;

}

void Object_Map::InsertHistoryBboxAndTwc(const Frame& CurrentFrame)
{
    unique_lock<mutex> lock(mMutex);
    mHistoryBbox[CurrentFrame.mTimeStamp] = mLastBbox;
    mHistoryTwc[CurrentFrame.mTimeStamp] = Converter::toMatrix4f(CurrentFrame.mTcw).inverse();

}

void Object_Map::AlignToCanonical()
{
    // time this function
    auto start = std::chrono::high_resolution_clock::now();

    // calculate object shape again
    CalculateObjectShape(true);

    if (!mConfig.isKnown)
    {
        // just expand bbox and return because the prior model is not known
        const float expandFactor = 1 + mConfig.bbox.expand;
        mShape.a1 *= expandFactor;
        mShape.a2 *= expandFactor;
        mShape.a3 *= expandFactor;
        mShape.a1 += mConfig.bbox.incrementX;
        mShape.a2 += mConfig.bbox.incrementY;
        mShape.a3 += mConfig.bbox.incrementZ;
        return;
    }

    // get the state of the object
    Eigen::Matrix4d Tow;
    vector<Eigen::Vector3d> vPos;
    if (mbMonocular)
    {
        Tow = mTobjw.to_homogeneous_matrix();
        for (const auto& pMP : mvpMapPoints)
        {
            if (pMP->isBad())
                continue;
            vPos.push_back(Converter::toVector3d(pMP->GetWorldPos()));
        }
    }
    else 
    {
        Tow = mFrozenTow;
        for (const auto& point : mFrozenCloud->points)
            vPos.push_back(Eigen::Vector3d(point.x, point.y, point.z));
    }
    const size_t numCloudPoints = vPos.size();

    // common variables
    float metric_resolution = mConfig.align.metricResolution; // for model-based
    float normalized_resolution = mConfig.align.normalizedResolution; // for density-based

    // object-based
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::octree::OctreePointCloud<pcl::PointXYZ> tree(100);

    // set the resolution based on alignment strategy
    if (mConfig.align.isDensityBased)
    {
        // use normalized resolution for density-based alignment
        metric_resolution = normalized_resolution * pow(mShape.a1 * mShape.a2 * mShape.a3 * 8, 1.0/3.0);
    }
    else
    {
        // use metric resolution for model-based alignment
        normalized_resolution = metric_resolution / pow(mShape.a1 * mShape.a2 * mShape.a3 * 8, 1.0/3.0);

        // get a sampled version of the meta mesh for completion score
        if (mConfig.symmetry.isReflectional)
        {
            pcl::copyPointCloud(*mPriorModel, *sampledCloud);
            Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(sampledCloud, numCloudPoints);
        }

        // make an octree to check the occupancy of the object
        tree.setResolution(normalized_resolution);
        tree.setInputCloud(mPriorModel);
        tree.addPointsFromInputCloud();
    }

    // align the object to the canonical pose if it is not a symmetric object
    Eigen::Matrix4d coarseTow = Tow;
    if (!mConfig.symmetry.isRotational)
    {
        // sample angles uniformly
        std::vector<double> angles;
        const size_t numAngles = mConfig.align.numSampleAngles;
        for (int i = 0; i < numAngles; i++)
            angles.push_back((i * 360.0/numAngles - 180) / 180.0 * M_PI);
        int scores[numAngles];

        pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
        for(int i = 0; i < numAngles; i++)
        {   
            // Determine the new object pose and extent
            Eigen::Matrix3d Rno = Converter::eulerAnglesToMatrix(angles[i]);
            Eigen::Matrix4d Tno = Eigen::Matrix4d::Identity();
            Tno.block(0,0,3,3) = Rno;
            Eigen::Matrix4d Tnw = Tno * Tow;
            Tnw.block(0,3,3,1) = Eigen::Vector3d::Zero();

            // get the bbox size
            const auto sizeTrans = GetSizeTransFromTransform(vPos, Tnw);
            const Eigen::Vector3d newBboxMax = sizeTrans.first;
            const Eigen::Vector3d newBboxMin = -newBboxMax;
            Tnw.block(0,3,3,1) = sizeTrans.second;

            // transform the point cloud to the object coordinate space
            object->clear();
            for(const auto& pos : vPos)
            {
                Eigen::Vector3d newPos = Tnw.block(0,0,3,3) * pos + Tnw.block(0,3,3,1);
                newPos = (newPos - newBboxMin).cwiseQuotient(newBboxMax - newBboxMin);
                if (newPos(0) < 0 || newPos(0) > 1 || newPos(1) < 0 || newPos(1) > 1 || newPos(2) < 0 || newPos(2) > 1)
                {
                    cout << "Point outside the bounding box" << endl;
                    continue;
                }
                object->emplace_back(newPos(0),newPos(1),newPos(2));
            }

            // compute the score
            if (mConfig.align.isDensityBased)
                scores[i] = ComputeDensityScore(object);
            else
            {
                scores[i] = ComputeOccupancyScoreOctree(tree, object);
                // also compute the completion for some classes with dense normalized meshes
                if (mConfig.symmetry.isReflectional)
                {
                    pcl::octree::OctreePointCloud<pcl::PointXYZ> objectTree(normalized_resolution);
                    objectTree.setInputCloud(object);
                    objectTree.addPointsFromInputCloud();
                    scores[i] += ComputeOccupancyScoreOctree(objectTree, sampledCloud);
                }
            }
        }

        // // [Debug] check the scores
        // if (mnClass == 56)
        // {
        //     std::cout << "All scores: " << std::endl;
        //     for(int i = 0; i < numAngles; i++)
        //         std::cout << scores[i] << std::endl;    
        // }
        
        // find the best alignment
        double bestAngle = 0;
        if (!mConfig.align.polyFit)
        {
            int best = 0;
            for(int i = 1; i < numAngles; i++)
            {
                if(scores[i] > scores[best])
                    best = i;
            }
            bestAngle = angles[best];
        }
        else
        {
            // fit a quadratic to the scores
            Eigen::VectorXd eAngles(numAngles);
            Eigen::VectorXd eScores(numAngles);
            for(int i = 0; i < numAngles; i++)
            {
                eAngles(i) = angles[i];
                eScores(i) = static_cast<double>(scores[i]);
            }
            PolyFit polyFit(eAngles, eScores, 2);
            bestAngle = polyFit.getExtrema();
        }
        cout << "Best angle chosen for class:  " << mnClass << " " << bestAngle << " " << bestAngle * 180.0 / M_PI << endl;

        // compute the new object pose and extent
        Eigen::Matrix3d Rno = Converter::eulerAnglesToMatrix(bestAngle);
        Eigen::Matrix4d Tno = Eigen::Matrix4d::Identity();
        Tno.block(0,0,3,3) = Rno;
        coarseTow = Tno * Tow;
    }

    // get the bbox size
    coarseTow.block(0,3,3,1) = Eigen::Vector3d::Zero();
    auto sizeTrans = GetSizeTransFromTransform(vPos, coarseTow);
    Eigen::Vector3d coarseBboxMax = sizeTrans.first;
    Eigen::Vector3d coarseBboxMin = -coarseBboxMax;
    coarseTow.block(0,3,3,1) = sizeTrans.second;

    // // [TODO] - update this with future implementation of partial observations
    // if (mnClass == 32)
    // {
    //     // A ball is a special case where the object is probably a sphere - easiest to cater to partial observations
    //     // [Note] - not all sports balls are spherical, but this is a good approximation
    //     float maxSize = std::max(std::max(coarseBboxMax(0), coarseBboxMax(1)), coarseBboxMax(2));
    //     coarseTow.block(0,3,3,1) += Eigen::Vector3d(maxSize, maxSize, maxSize) - coarseBboxMax;
    //     coarseBboxMax = Eigen::Vector3d(maxSize, maxSize, maxSize);
    //     coarseBboxMin = -coarseBboxMax;
    // }

    // output the time taken
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Alignment took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    // once a coarse alignment has been found, refine the alignment using ICP
    Eigen::Matrix4d fineTow = coarseTow;
    Eigen::Vector3d fineBboxMax = coarseBboxMax;
    if (mConfig.icp.enabled)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr prior_aligned(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& pos : vPos)
        {
            Eigen::Vector3d newPos = coarseTow.block(0,0,3,3) * pos + coarseTow.block(0,3,3,1);
            object_aligned->emplace_back(newPos(0),newPos(1),newPos(2));
        }
        
        // the canonical prior object in the object frame
        for(const auto& point : mPriorModel->points)
        {
            Eigen::Vector3d pos = {point.x, point.y, point.z};
            pos = pos.cwiseProduct(coarseBboxMax - coarseBboxMin) + coarseBboxMin;
            prior_aligned->emplace_back(pos(0),pos(1),pos(2));
        }

        // // [Debug] - check the accuracy before alignment
        // pcl::octree::OctreePointCloud<pcl::PointXYZ> prior_tree(metric_resolution);
        // prior_tree.setInputCloud(prior_aligned);
        // prior_tree.addPointsFromInputCloud();
        // std::cout << "Prior accuracy: " << ComputeOccupancyScoreOctree(prior_tree, object_aligned) << std::endl;

        // the ICP algorithm - constrained to be 4 dof - 3 dof translation and rotation about the z-axis
        start = std::chrono::high_resolution_clock::now();
        pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
        pcl::registration::WarpPointRigid4D<pcl::PointXYZ, pcl::PointXYZ>::Ptr warp_fcn 
            (new pcl::registration::WarpPointRigid4D<pcl::PointXYZ, pcl::PointXYZ>);
        pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>::Ptr te 
            (new pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>);
        te->setWarpFunction (warp_fcn);
        icp.setTransformationEstimation(te);
        icp.setMaximumIterations(mConfig.icp.maxIterations);
        icp.setInputSource(object_aligned);
        icp.setInputTarget(prior_aligned);

        // [Debug] - check the accuracy after alignment
        icp.align(*object_aligned);
        std::cout << icp.getFinalTransformation() << std::endl;
        end = std::chrono::high_resolution_clock::now();

        // // [Debug] - check the accuracy after alignment
        // std::cout << "Prior accuracy: " << ComputeOccupancyScoreOctree(prior_tree, object_aligned) << std::endl;
        std::cout << "ICP took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

        // if transformation is too large, reject the alignment
        // check translation against the diagonal of the bounding box and rotation
        Eigen::Matrix4d refinement = icp.getFinalTransformation().cast<double>();
        if(refinement.block(0,3,3,1).norm() > mConfig.icp.maxTransPercent * (coarseBboxMax - coarseBboxMin).norm())
        {
            std::cout << "ICP output rejected because of large translation" << std::endl;
            refinement = Eigen::Matrix4d::Identity();
        }
        else if (refinement(0,0) < mConfig.icp.minRotCos)
        {
            std::cout << "ICP output rejected because of large rotation" << std::endl;
            refinement = Eigen::Matrix4d::Identity();
        }
        fineTow = refinement.inverse() * coarseTow;
        fineTow.block(0,3,3,1) = Eigen::Vector3d::Zero();
        auto sizeTrans = GetSizeTransFromTransform(vPos, fineTow);
        fineBboxMax = sizeTrans.first;
        fineBboxMax += refinement.block(0,3,3,1).cwiseAbs();
        fineTow.block(0,3,3,1) = sizeTrans.second;
    }

    // apply the best alignment
    mShape.mTobjw = SE3Quat(fineTow.block(0,0,3,3), fineTow.block(0,3,3,1));
    mShape.a1 = fineBboxMax(0);
    mShape.a2 = fineBboxMax(1);
    mShape.a3 = fineBboxMax(2);
    mShape.mfMaxDist = sqrt(mShape.a1 * mShape.a1 + mShape.a2 * mShape.a2 + mShape.a3 * mShape.a3);
    mTobjw = mShape.mTobjw;
}

int Object_Map::ComputeOccupancyScoreOctree(const pcl::octree::OctreePointCloud<pcl::PointXYZ>& octree, 
                                            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const
{
    int score = 0;
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        if (octree.isVoxelOccupiedAtPoint(cloud->points[i]))
            score++;
        else
            score--;
    }
    return score;
}

int Object_Map::ComputeDensityScore(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) const
{
    omp_set_dynamic(0);
    double energy = 0;
    #pragma omp parallel for reduction(+:energy) num_threads(8)
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto& point = cloud->points[i];
        if (point.x > 1 || point.x < 0 || point.y > 1 || point.y < 0 || point.z > 1 || point.z < 0)
            std::cout << "Something went wrong" << std::endl;

        // trilinear interpolation
        int ix0 = floor((GRID_SIZE-1) * point.x);
        int iy0 = floor((GRID_SIZE-1) * point.y);
        int iz0 = floor((GRID_SIZE-1) * point.z);
        
        // need to clamp because of floating point errors
        ix0 = max(0, min(ix0, GRID_SIZE - 1));
        iy0 = max(0, min(iy0, GRID_SIZE - 1));
        iz0 = max(0, min(iz0, GRID_SIZE - 1));

        const int ix1 = std::min(ix0 + 1, GRID_SIZE - 1);
        const int iy1 = std::min(iy0 + 1, GRID_SIZE - 1);
        const int iz1 = std::min(iz0 + 1, GRID_SIZE - 1);
        const float xd = GRID_SIZE * point.x - ix0;
        const float yd = GRID_SIZE * point.y - iy0;
        const float zd = GRID_SIZE * point.z - iz0;
        const float c00 = mPriorDensity[ix0 + GRID_SIZE * iy0 + GRID_SIZE_SQ * iz0] * (1 - xd) + mPriorDensity[ix1 + GRID_SIZE * iy0 + GRID_SIZE_SQ * iz0] * xd;
        const float c10 = mPriorDensity[ix0 + GRID_SIZE * iy1 + GRID_SIZE_SQ * iz0] * (1 - xd) + mPriorDensity[ix1 + GRID_SIZE * iy1 + GRID_SIZE_SQ * iz0] * xd;
        const float c01 = mPriorDensity[ix0 + GRID_SIZE * iy0 + GRID_SIZE_SQ * iz1] * (1 - xd) + mPriorDensity[ix1 + GRID_SIZE * iy0 + GRID_SIZE_SQ * iz1] * xd;
        const float c11 = mPriorDensity[ix0 + GRID_SIZE * iy1 + GRID_SIZE_SQ * iz1] * (1 - xd) + mPriorDensity[ix1 + GRID_SIZE * iy1 + GRID_SIZE_SQ * iz1] * xd;
        const float c0 = c00 * (1 - yd) + c10 * yd;
        const float c1 = c01 * (1 - yd) + c11 * yd;
        const float c = c0 * (1 - zd) + c1 * zd;
        energy += c;
    }
    return static_cast<int>(energy * 10);
}

pair<Eigen::Vector3d, Eigen::Vector3d> Object_Map::GetSizeTransFromTransform(const vector<Eigen::Vector3d>& points, const Eigen::Matrix4d& T)
{
    // transform the point cloud to the object coordinate space
    Eigen::Vector3d bboxMax;
    const size_t numCloudPoints = points.size();
    Eigen::VectorXd xValues(numCloudPoints), yValues(numCloudPoints), zValues(numCloudPoints);
    for (size_t i = 0; i < numCloudPoints; i++)
    {
        Eigen::Vector3d newPos = T.block(0,0,3,3) * points[i] + T.block(0,3,3,1);
        xValues(i) = newPos(0);
        yValues(i) = newPos(1);
        zValues(i) = newPos(2);
    }
    const Eigen::Vector3d maxVals = {xValues.maxCoeff(), yValues.maxCoeff(), zValues.maxCoeff()};
    const Eigen::Vector3d minVals = {xValues.minCoeff(), yValues.minCoeff(), zValues.minCoeff()};
    Eigen::Vector3d size = maxVals - minVals;
    size = size.cwiseAbs();
    size *= (1 + mConfig.bbox.expand);
    bboxMax = size/2.0;

    // extra padding in the z direction as the objects are attached to something
    bboxMax(0) += mConfig.bbox.incrementX;
    bboxMax(1) += mConfig.bbox.incrementY;
    bboxMax(2) += mConfig.bbox.incrementZ;

    // also return the translation as the center of the bounding box
    Eigen::Vector3d translation = -(maxVals + minVals) / 2;
    return make_pair(bboxMax, translation);
}

void Object_Map::AddToCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& Twc)
{
    unique_lock<mutex> lock(mMutexCloud);
    if (haveNeRF || cloud->points.size() == 0)
        return;

    const size_t minCloudPoints = mConfig.pointcloud.minPoints;
    const size_t maxCloudPoints = mConfig.pointcloud.maxPoints;

    // auto start = std::chrono::high_resolution_clock::now();
    // downsample the cloud - only get the most recurring points
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled;
    downsampled = Utils::pointcloudDownsample<pcl::PointXYZ>(cloud,
                                                             mConfig.downsample.voxelSize,
                                                             mConfig.downsample.minPointsPerVoxel);
    if (downsampled->points.size() < minCloudPoints)
        return;
    
    if (mConfig.outlierRemoval.enabled)
    {
        downsampled = Utils::pointcloudOutlierRemoval<pcl::PointXYZ>(downsampled, 
                                                                 mConfig.outlierRemoval.minNeighbors,
                                                                 mConfig.outlierRemoval.stdDev);
        if (downsampled->points.size() < minCloudPoints)
            return;
    }

    // Check for clusters
    if (mConfig.clustering.enabled)
    {
        vector<pcl::PointIndices> clusterIndices;
        Utils::pointcloudEuclideanClustering(downsampled, clusterIndices, mConfig.clustering.tolerance);
        if (clusterIndices.size() > 1)
        {
            // cout << "Multiple clusters detected" << endl;
            // cout << "Number of clusters: " << clusterIndices.size() << endl;
            
            // add clusters that are big enough and pass tests if they are enabled
            pcl::PointCloud<pcl::PointXYZ>::Ptr remainder(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& indices : clusterIndices)
            {
                const size_t clusterSize = indices.indices.size();
                if (clusterSize > minCloudPoints)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    if (mConfig.centroidTTest.enabled || mConfig.rankSumTest.enabled)
                    {
                        // transform the cluster to the global frame
                        for (const auto& index : indices.indices)
                            tempCloud->points.push_back(downsampled->points[index]);
                        pcl::transformPointCloud(*tempCloud, *tempCloud, Twc);
                    }
                    
                    if (mConfig.centroidTTest.enabled)
                    {
                        // push it to a queue to check later
                        mvCloudClustersToCheck.push_back(tempCloud);
                        continue;
                    }
                    
                    if (mConfig.rankSumTest.enabled)
                    {
                        if (mCloud->points.size() > minCloudPoints && mnObs > mConfig.rankSumTest.minHistorySize)
                        {
                            // transform the cluster to the global frame
                            pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
                            for (const auto& index : indices.indices)
                                tempCloud->points.push_back(downsampled->points[index]);
                            pcl::transformPointCloud(*tempCloud, *tempCloud, Twc);
                            for (size_t i=0; i<mCloud->points.size(); i++)
                                tempCloud->points.push_back(mCloud->points[i]);
                            Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(tempCloud, maxCloudPoints);
                            if (!Utils::checkRankSumTest(mCloud, tempCloud))
                                continue;
                        }
                    }
                    
                    for (const auto& index : indices.indices)
                        remainder->points.push_back(downsampled->points[index]);
                }
            }
            downsampled = remainder;
        }
        if (downsampled->points.size() < minCloudPoints)
            return;
    }

    // transform to get the cloud in global
    pcl::transformPointCloud(*downsampled, *downsampled, Twc);

    // Make a temporary cloud with the new points for testing
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : downsampled->points)
        tempCloud->points.push_back(point);
    for (const auto& point : mCloud->points)
        tempCloud->points.push_back(point);
    Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(tempCloud, mConfig.pointcloud.maxPoints);

    // Wilcoxn Rank Sum Test
    if (mConfig.rankSumTest.enabled && mCloud->points.size() > minCloudPoints && mnObs > mConfig.rankSumTest.minHistorySize)
    {
        if (!Utils::checkRankSumTest(mCloud, tempCloud))
        {
            cout << "Wilcoxn Rank Sum Test failed" << endl;
            return;
        }
    }
    
    // // t-test on the centroid 
    if (mConfig.centroidTTest.enabled)
    {
        if (mvCloudCentroidHistory.size() >= mConfig.centroidTTest.minHistorySize)
        {
            if (!CentroidTTest(tempCloud))
                return;

            if (mvCloudClustersToCheck.size() > 0)
            {
                bool somePassed = false;
                for (const auto& cluster : mvCloudClustersToCheck)
                {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr tempTempCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    for (const auto& point : cluster->points)
                        tempTempCloud->points.push_back(point);
                    for (const auto& point : tempCloud->points)
                        tempTempCloud->points.push_back(point);
                    Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(tempTempCloud, maxCloudPoints);
                    if (!CentroidTTest(tempTempCloud, true))
                        continue;
                    somePassed = true;
                    for (const auto& point : cluster->points)
                        tempCloud->points.push_back(point);
                }
                mvCloudClustersToCheck.clear();
                if (somePassed)
                    Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(tempCloud, maxCloudPoints);
            }
        }

        // add the centroid to the history
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& point : tempCloud->points)
            centroid += Eigen::Vector3d(point.x, point.y, point.z);
        centroid /= tempCloud->points.size();
        AddCloudCentroidToHistory(centroid);
    }

    // everything passed, replace the cloud
    mCloud = tempCloud;

    // auto end = std::chrono::high_resolution_clock::now();
    // cout << "Adding to cloud took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;
}


void Object_Map::AddCloudCentroidToHistory(const Eigen::Vector3d& centroid)
{
    const size_t historySize = mvCloudCentroidHistory.size();
    mvCloudCentroidHistory.push_back(centroid);

    // recompute the mean
    mCloudCentroidHistoryMean *= historySize;
    mCloudCentroidHistoryMean += centroid;
    mCloudCentroidHistoryMean /= (historySize + 1);

    // recompute the standard deviation
    mCloudCentroidHistoryStdDev = Eigen::Vector3d::Zero();
    for (size_t i=0; i<historySize+1; i++)
    {
        const Eigen::Vector3d diff = mvCloudCentroidHistory[i] - mCloudCentroidHistoryMean;
        mCloudCentroidHistoryStdDev += diff.cwiseProduct(diff);
    }
    mCloudCentroidHistoryStdDev /= (historySize + 1);
    mCloudCentroidHistoryStdDev = mCloudCentroidHistoryStdDev.cwiseSqrt();

    // cout << "Centroid history mean: " << mCloudCentroidHistoryMean.transpose() << endl;
    // cout << "Centroid history std dev: " << mCloudCentroidHistoryStdDev.transpose() << endl;
}

bool Object_Map::CentroidTTest(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const bool addToHistory)
{
    // compute centroid 
    const size_t clusterSize = cloud->points.size();
    if (clusterSize == 0)
        return false;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& point : cloud->points)
        centroid += Eigen::Vector3d(point.x, point.y, point.z);
    centroid /= clusterSize;

    // compute the t-statistic
    const int numHistory = mvCloudCentroidHistory.size();
    Eigen::Vector3d tStat = centroid - mCloudCentroidHistoryMean;
    tStat = tStat.cwiseAbs() * sqrt(numHistory);
    tStat = tStat.cwiseQuotient(mCloudCentroidHistoryStdDev);
    
    const int dof = min(100, numHistory - 1);
    const float thresh = mTTest[dof][3]; // significance level
    if ((tStat(0) + tStat(1) + tStat(2)) > 10*thresh)
    {
        cout << "Cluster is bad" << endl;
        return false;
    }
    cout << "Cluser is good" << endl;
    if (addToHistory)
        AddCloudCentroidToHistory(centroid);
    return true;
}


vector<Eigen::Vector3f> Object_Map::GetCloudPoints() const
{
    vector<Eigen::Vector3f> points;
    if (mbMonocular) {
        // give the map points
        unique_lock<mutex> lock(mMutexMapPoints);
        for (const auto& point : mvpMapPoints) {
            Eigen::Vector3d pos = Converter::toVector3d(point->GetWorldPos());
            points.emplace_back(pos(0), pos(1), pos(2));
        }
    }
    else {
        // give the cloud points
        unique_lock<mutex> lock(mMutexCloud);
        for (const auto& point : mCloud->points)
            points.emplace_back(point.x, point.y, point.z);
    }
    return points;
}

} // namespace ORB_SLAM2