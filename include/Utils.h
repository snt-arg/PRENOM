/**
 * 🚀 [vS-Graphs] A class for keeping the utility functions
 */

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cassert>
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

#include <unsupported/Eigen/Splines>
#include <unordered_set>

#include "KDLineTree.h"
#include "Point.h"

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

        /**
         * @brief Samples points from the pointcloud using the farthest point sampling method
         *
         * @param cloud the pointcloud to be sampled
         * @param numSamples the number of samples to be taken
         */
        template <typename PointT>
        static void pointcloudFarthestPointSampling(typename pcl::PointCloud<PointT>::Ptr &cloud, size_t numSamples);

        /**
         * @brief Clusters the pointclouds based on the given cluster tolerance
         *
         * @param cloud the pointcloud to be clustered
         * @param clusterIndices the indices of the output clusters
         * @param clusterTolerance the tolerance for clustering
         */
        static void pointcloudEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<pcl::PointIndices> &clusterIndices, const float clusterTolerance);


        /**
         * @brief Gets the robust size from the pointcloud
         *
         * @param cloud the pointcloud to be processed
         * @param kneedleSensitivity the sensitivity for the kneedle method
         * @param numUncertain the number of uncertain points
         */
        template <typename PointT>
        static tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> getRobustSizeFromPointCloud(const typename pcl::PointCloud<PointT>::Ptr &cloud,
                                                                                                    const float kneedleSensitivity, const size_t numUncertain);
        static Eigen::VectorXd getKneedleThreshold(const Eigen::VectorXd& localMaximasY, const Eigen::VectorXd& xNorm, const float sensistivity);
        static int getKneePoint(const Eigen::VectorXd &values, const float sensitivity = 1.0, const uint8_t polyDeg = 0);
        static Eigen::VectorXd minMaxNormalize(const Eigen::VectorXd& values);
        static Eigen::VectorXi findLocalExtrema(const Eigen::VectorXd& values, bool max);

        /**
         * @brief Checks the rank sum test for the incoming pointcloud
         *
         * @param existingCloud the existing pointcloud
         * @param incomingCloud the incoming pointcloud
         */
        static bool checkRankSumTest(const pcl::PointCloud<pcl::PointXYZ>::Ptr &existingCloud, 
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &incomingCloud);

    };


    class PolyFit {
    public:
        PolyFit(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const int degree)
            : x_(x), y_(y), degree(degree)
        {
            Eigen::MatrixXd xs(x.size(), degree + 1);
            xs.col(0).setOnes();
            for (int i = 1; i <= degree; ++i) {
                xs.col(i).array() = xs.col(i - 1).array() * x_.array();
            }

            result_.resize(degree + 1);
            auto result_map = Eigen::Map<Eigen::VectorXd>(result_.data(), result_.size());

            auto decomposition = xs.householderQr();
            result_map = decomposition.solve(y_);
        }

        double operator()(double x) const {
            double result = 0;
            for (int i = 0; i <= degree; ++i) {
                result += result_[i] * std::pow(x, i);
            }
            return result;
        }

        int degree;

        double getExtrema() const {
            // only for degree 2 polynomials
            assert(degree == 2);
            return -result_[1] / (2 * result_[2]);
        }

    private:
        Eigen::VectorXd x_;
        Eigen::VectorXd y_;
        std::vector<double> result_;
    };
}

#endif // UTILS_H