/*
* Modification: PRENOM
* Version: 1.0
* Created: 12/25/2024
* Author: Saad Ejaz
*/

#include "Utils.h"

namespace ORB_SLAM2
{

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr Utils::pointcloudDownsample(
    const typename pcl::PointCloud<PointT>::Ptr &cloud, const float leafSize, const unsigned int minPointsPerVoxel)
{
    // The filtered point cloud object
    typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>());

    // Define the downsampling filter
    typename pcl::VoxelGrid<PointT>::Ptr downsampleFilter(new pcl::VoxelGrid<PointT>());

    // Set the parameters of the downsampling filter
    downsampleFilter->setLeafSize(leafSize, leafSize, leafSize);
    downsampleFilter->setMinimumPointsNumberPerVoxel(minPointsPerVoxel);
    downsampleFilter->setInputCloud(cloud);

    // Apply the downsampling filter
    downsampleFilter->filter(*filteredCloud);
    filteredCloud->header = cloud->header;

    return filteredCloud;
}
template pcl::PointCloud<pcl::PointXYZ>::Ptr Utils::pointcloudDownsample<pcl::PointXYZ>(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const float, const unsigned int);

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr Utils::pointcloudDistanceFilter(
    const typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    // Variables
    double distance;
    const pair<float, float> thresholds = make_pair(0.3, 3.5);
    const float thresholdNear = thresholds.first;
    const float thresholdFar = thresholds.second;

    // Define the filtered point cloud object
    typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>());
    filteredCloud->reserve(cloud->size());

    // Filter the point cloud
    copy_if(cloud->begin(),
            cloud->end(),
            back_inserter(filteredCloud->points),
            [&](const PointT &p)
            {
                distance = p.z;
                return distance > thresholdNear && distance < thresholdFar;
            });

    filteredCloud->height = 1;
    filteredCloud->is_dense = false;
    filteredCloud->header = cloud->header;
    filteredCloud->width = filteredCloud->size();

    return filteredCloud;
}
template pcl::PointCloud<pcl::PointXYZ>::Ptr Utils::pointcloudDistanceFilter<pcl::PointXYZ>(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &);


void Utils::pointcloudEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<pcl::PointIndices> &clusterIndices, const float clusterTolerance)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(1e6);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);
}


template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr Utils::pointcloudOutlierRemoval(
    const typename pcl::PointCloud<PointT>::Ptr &cloud, const int meanThresh, const float stdDevThresh)
{
    // Check if the input cloud is empty
    if (cloud->points.size() == 0)
        return cloud;

    // Create a container for the filtered cloud
    typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>);

    // Create the filtering object: StatisticalOutlierRemoval
    pcl::StatisticalOutlierRemoval<PointT> outlierRemoval;
    outlierRemoval.setInputCloud(cloud);
    outlierRemoval.setMeanK(meanThresh);
    outlierRemoval.setStddevMulThresh(stdDevThresh);
    outlierRemoval.filter(*filteredCloud);
    return filteredCloud;
}
template pcl::PointCloud<pcl::PointXYZ>::Ptr Utils::pointcloudOutlierRemoval<pcl::PointXYZ>(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const int, const float);

template <typename PointT>
void Utils::pointcloudFarthestPointSampling(typename pcl::PointCloud<PointT>::Ptr &cloud, size_t numSamples)
{
    if (cloud->size() <= numSamples)
        numSamples = cloud->size();

    vector<Point> pointData;
    int count = 0;
    for (const auto &point : cloud->points)
        pointData.emplace_back(Point(point.x, point.y, point.z, 1 << 30, count++));

    const int pointSize = pointData.size();
    auto points = (Point *)malloc(pointSize * sizeof(Point));
    auto samplePoints = (Point *)malloc(numSamples * sizeof(Point));
    for (int i = 0; i < pointSize; i++)
        points[i] = pointData[i];

    KDLineTree tree = KDLineTree(points, pointSize, 3, samplePoints);
    Point initPoint = points[0];
    samplePoints[0] = initPoint;
    tree.buildKDtree();
    tree.init(initPoint);
    tree.sample(numSamples);

    cloud->clear();
    for (size_t i = 0; i < numSamples; i++)
    {
        Point samplePoint = samplePoints[i];
        cloud->emplace_back(PointT(samplePoint.pos[0], samplePoint.pos[1], samplePoint.pos[2]));
    }

    free(points);
    free(samplePoints);
}
template void Utils::pointcloudFarthestPointSampling<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>::Ptr &, size_t);


bool Utils::checkRankSumTest(const pcl::PointCloud<pcl::PointXYZ>::Ptr &existingCloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr &incomingCloud)
{
    const size_t numExisting = existingCloud->size();
    const size_t numIncoming = incomingCloud->size();
    const double criticalValue = 2.576; // 5% significance level two-tailed test
    double cumulativeZScore = 0;

    for (int k=0; k<3; k++)
    {
        // sort the values while keeping track of which cloud they belong to
        vector<pair<double, bool>> values;
        for (size_t i = 0; i < numExisting; i++)
            values.push_back(make_pair(existingCloud->points[i].data[k], true));
        for (size_t i = 0; i < numIncoming; i++)
            values.push_back(make_pair(incomingCloud->points[i].data[k], false));

        // sort the values
        sort(values.begin(), values.end(), [](const pair<double, bool> &a, const pair<double, bool> &b) { return a.first < b.first; });

        // compute the rank sum - since values are continuous, we can ignore ties
        double rankSumExisting = 0;
        double rankSumIncoming = 0;
        for (size_t i = 0; i < values.size(); i++)
        {
            if (values[i].second)
                rankSumExisting += i+1;
            else
                rankSumIncoming += i+1;
        }
        rankSumExisting -= numExisting * (numExisting + 1) / 2;
        rankSumIncoming -= numIncoming * (numIncoming + 1) / 2;

        // choose the smaller rank sum
        const double rankSum = min(rankSumExisting, rankSumIncoming);
        const double mean = numExisting * numIncoming / 2;
        const double stdDev = sqrt(numExisting * numIncoming * (numExisting + numIncoming + 1) / 12);
        const double zScore = (rankSum - mean) / stdDev;
        // if (fabs(zScore) > criticalValue)
        //     return false;
        cumulativeZScore += zScore;
    }
    if (cumulativeZScore > 7*criticalValue)
        return false;
    return true;
}

} // namespace ORB_SLAM2