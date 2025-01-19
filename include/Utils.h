/**
 * 🚀 [vS-Graphs] A class for keeping the utility functions
 */

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace ORB_SLAM2
{
    class Utils
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Variables
        static constexpr double DEG_TO_RAD = M_PI / 180.0;

        /**
         * @brief Downsamples the pointclouds based on the given leaf size
         *
         * @param cloud the pointcloud to be downsampled
         * @param leafSize the leaf size for downsampling
         */
        template <typename PointT>
        static typename pcl::PointCloud<PointT>::Ptr pointcloudDownsample(
            const typename pcl::PointCloud<PointT>::Ptr &cloud, const float leafSize, const unsigned int minPointsPerVoxel);


        /**
         * @brief Filters the pointclouds based on the given min/max distance acceptable
         *
         * @param cloud the pointcloud to be filtered
         */
        template <typename PointT>
        static typename pcl::PointCloud<PointT>::Ptr pointcloudDistanceFilter(
            const typename pcl::PointCloud<PointT>::Ptr &cloud);

        /**
         * @brief Removes the points that are farther away from their neighbors
         *
         * @param cloud the pointcloud to be filtered
         * @param meanThresh the mean threshold for neighbor points
         * @param stdDevThresh the standard deviation threshold for neighbor points
         */
        template <typename PointT>
        static typename pcl::PointCloud<PointT>::Ptr pointcloudOutlierRemoval(
            const typename pcl::PointCloud<PointT>::Ptr &cloud, const int meanThresh, const float stdDevThresh);
    };
}

#endif // UTILS_H