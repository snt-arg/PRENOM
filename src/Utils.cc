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
        const std::pair<float, float> thresholds = std::make_pair(0.3, 3.5);
        const float thresholdNear = thresholds.first;
        const float thresholdFar = thresholds.second;

        // Define the filtered point cloud object
        typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>());
        filteredCloud->reserve(cloud->size());

        // Filter the point cloud
        std::copy_if(cloud->begin(),
                     cloud->end(),
                     std::back_inserter(filteredCloud->points),
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

        // Return the filtered cloud
        return filteredCloud;
    }
    template pcl::PointCloud<pcl::PointXYZ>::Ptr Utils::pointcloudOutlierRemoval<pcl::PointXYZ>(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &, const int, const float);

} // namespace ORB_SLAM2