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

    // Return the filtered cloud
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

template <typename PointT>
tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> Utils::getRobustSizeFromPointCloud(const typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    // use the knee method to rule out outliers
    const size_t numPoints = cloud->size();
    const float kneedleSensitivity = 20.0;
    size_t numUncertain = static_cast<size_t>(numPoints * 0.020);
    if (numUncertain < 20)
        numUncertain = 20;

    // sort each axis
    Eigen::VectorXd xValues(numPoints), yValues(numPoints), zValues(numPoints);
    for (size_t i=0; i<numPoints; i++)
    {
        xValues(i) = cloud->points[i].x;
        yValues(i) = cloud->points[i].y;
        zValues(i) = cloud->points[i].z;
    }
    sort(xValues.begin(), xValues.end());
    sort(yValues.begin(), yValues.end());
    sort(zValues.begin(), zValues.end());

    // if not enough points, return the values
    // if (numPoints < 200)
        return make_tuple(xValues, yValues, zValues);

    // for each dimenstion, calculate the size for all points until the numUncertain
    Eigen::VectorXd xPosSizes(numUncertain), xNegSizes(numUncertain);
    Eigen::VectorXd yPosSizes(numUncertain), yNegSizes(numUncertain);
    Eigen::VectorXd zPosSizes(numUncertain), zNegSizes(numUncertain);
    for (size_t i=0; i<numUncertain; i++) {
        xPosSizes(i) = xValues(numPoints - 1) - xValues(i);
        xNegSizes(i) = xValues(numPoints - i - 1) - xValues(0);
        yPosSizes(i) = yValues(numPoints - 1) - yValues(i);
        yNegSizes(i) = yValues(numPoints - i - 1) - yValues(0);
        zPosSizes(i) = zValues(numPoints - 1) - zValues(i);
        zNegSizes(i) = zValues(numPoints - i - 1) - zValues(0);
    }

    // use the knee method to rule out outliers
    int xKneePos = getKneePoint(xPosSizes, kneedleSensitivity, 3) + 1;
    int xKneeNeg = getKneePoint(xNegSizes, kneedleSensitivity, 3) + 1;
    int yKneePos = getKneePoint(yPosSizes, kneedleSensitivity, 3) + 1;
    int yKneeNeg = getKneePoint(yNegSizes, kneedleSensitivity, 3) + 1;
    int zKneePos = getKneePoint(zPosSizes, kneedleSensitivity, 3) + 1;
    int zKneeNeg = getKneePoint(zNegSizes, kneedleSensitivity, 3) + 1;

    if (xKneePos > 30)
    {
        cout << "xKneePos: " << xKneePos << endl;
        cout << xPosSizes << endl;
        cout << endl;
    }

    // xKneePos = xKneePos == -1 ? 0 : xKneePos;
    // xKneeNeg = xKneeNeg == -1 ? 0 : xKneeNeg;
    // yKneePos = yKneePos == -1 ? 0 : yKneePos;
    // yKneeNeg = yKneeNeg == -1 ? 0 : yKneeNeg;
    // zKneePos = zKneePos == -1 ? 0 : zKneePos;
    // zKneeNeg = zKneeNeg == -1 ? 0 : zKneeNeg;

    // most conservative knee point
    const double minX = xValues(xKneePos);
    const double minY = yValues(yKneePos);
    const double minZ = zValues(zKneePos);
    const double maxX = xValues(numPoints - xKneeNeg - 1);
    const double maxY = yValues(numPoints - yKneeNeg - 1);
    const double maxZ = zValues(numPoints - zKneeNeg - 1);

    cout << "Knee points: " << xKneePos << " " << xKneeNeg << " " << yKneePos << " " << yKneeNeg << " " << zKneePos << " " << zKneeNeg << endl;

    typename pcl::PointCloud<PointT>::Ptr filteredCloud(new pcl::PointCloud<PointT>());
    for (size_t i=0; i<numPoints; i++)
    {
        if (cloud->points[i].x >= minX && cloud->points[i].x <= maxX &&
            cloud->points[i].y >= minY && cloud->points[i].y <= maxY &&
            cloud->points[i].z >= minZ && cloud->points[i].z <= maxZ)
            filteredCloud->push_back(cloud->points[i]);
    }
    const size_t numAfterFilter = filteredCloud->size();
    cout << "Before: " << numPoints << " After: " << numAfterFilter << endl;

    // return the filtered cloud as x,y,z values
    Eigen::VectorXd xValuesFiltered(numAfterFilter), yValuesFiltered(numAfterFilter), zValuesFiltered(numAfterFilter);
    for (size_t i=0; i<numAfterFilter; i++)
    {
        xValuesFiltered(i) = filteredCloud->points[i].x;
        yValuesFiltered(i) = filteredCloud->points[i].y;
        zValuesFiltered(i) = filteredCloud->points[i].z;
    }
    sort(xValuesFiltered.begin(), xValuesFiltered.end());
    sort(yValuesFiltered.begin(), yValuesFiltered.end());
    sort(zValuesFiltered.begin(), zValuesFiltered.end());

    return make_tuple(xValuesFiltered, yValuesFiltered, zValuesFiltered);
}
template tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> Utils::getRobustSizeFromPointCloud<pcl::PointXYZ>(const pcl::PointCloud<pcl::PointXYZ>::Ptr &);


int Utils::getKneePoint(const Eigen::VectorXd &values, const float sensitivity, const uint8_t polyDeg)
{
    // Use the Kneedle algorithm to find the knee point
    // The algorithm is based on the paper "Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior" by Satopaa et al.
    const size_t numPoints = values.size();
    
    // the x values are the indices of the values - fit the spline
    const Eigen::VectorXi xVec = Eigen::VectorXi::LinSpaced(numPoints, 0, numPoints-1);
    const Eigen::VectorXd xVecDouble = xVec.cast<double>();
    Eigen::VectorXd polyValues = values;

    // fit polynomial
    if (polyDeg != 0)
    {
        PolyFit polyFit(xVecDouble, values, polyDeg);
        for (int i = 0; i < numPoints; i++)
            polyValues(i) = polyFit(i);
    }

    Eigen::VectorXd xNorm = Utils::minMaxNormalize(xVecDouble);
    Eigen::VectorXd yNorm = Utils::minMaxNormalize(polyValues);
    Eigen::VectorXd yDiff(numPoints);
    Eigen::VectorXd xDiff(xNorm);

    // convex decreasing curve
    yNorm = Eigen::VectorXd::Ones(numPoints) - yNorm;
    yDiff = yNorm - xNorm;

    // find local maximas
    Eigen::VectorXi maximasIdx = Utils::findLocalExtrema(yDiff, true);
    const size_t numMaximas = maximasIdx.size();
    if (numMaximas == 0)
        return -1;
    Eigen::VectorXd maximasYDiff(numMaximas);
    for (size_t i=0; i<numMaximas; i++)
        maximasYDiff(i) = yDiff(maximasIdx(i));

    // final local minimas
    const Eigen::VectorXi minimasIdx = Utils::findLocalExtrema(yDiff, false);

    // compute the kneedle threshold
    Eigen::VectorXd kneedleThreshold = Utils::getKneedleThreshold(maximasYDiff, xNorm, sensitivity);

    // find the knee
    size_t curMaximaIdx = 0;
    size_t curMinimaIdx = 0;
    double threshold;
    size_t thresholdIdx;
    unordered_set<size_t> minimasIdxSet(minimasIdx.begin(), minimasIdx.end());
    unordered_set<size_t> maximasIdxSet(maximasIdx.begin(), maximasIdx.end());
    int knee = -1;
    for (int i=0; i<xDiff.size()-1; i++)
    {
        if (i < maximasIdx(0))
            continue;

        if (maximasIdxSet.find(i) != maximasIdxSet.end())
        {
            threshold = kneedleThreshold(curMaximaIdx);
            thresholdIdx = i;
            curMaximaIdx++;
        }

        if (minimasIdxSet.find(i) != minimasIdxSet.end())
        {
            threshold = 0.0;
            curMinimaIdx++;
        }

        if (yDiff(i+1) < threshold)
            knee = xVec(thresholdIdx);
    }

    // the kneedle method really wants to find a knee - check if it's significant enough
    if (knee>-1)
    {
        // knee is always < numPoints - 1
        const float significantThreshold = 0.10;
        const double prevDiff = fabs(values(knee+1) - values(0))/values(0);
        if (prevDiff < significantThreshold)
            knee = -1;
    }
    return knee;
}


Eigen::VectorXd Utils::getKneedleThreshold(const Eigen::VectorXd& maximasYDiff, const Eigen::VectorXd& xNorm, const float sensitivity)
{
    // calculate the scaled difference mean
    double scaledDiffMean = 0;
    const size_t xNormSize = xNorm.size();
    for (size_t i=0; i<xNormSize-1; i++)
        scaledDiffMean += xNorm(i+1) - xNorm(i);
    scaledDiffMean /= (xNormSize - 1);
    scaledDiffMean = sensitivity * fabs(scaledDiffMean);
    
    // calculate the threshold
    const Eigen::VectorXd output = maximasYDiff.array() - scaledDiffMean;
    return output;
}


Eigen::VectorXd Utils::minMaxNormalize(const Eigen::VectorXd &values)
{
    // get the min and max values
    const double minValue = values.minCoeff();
    const double diff = values.maxCoeff() - minValue;

    // normalize the values
    const Eigen::VectorXd normalizedValues = (values.array() - minValue) / diff;
    return normalizedValues;
}

Eigen::VectorXi Utils::findLocalExtrema(const Eigen::VectorXd &values, bool max)
{
    vector<int> extrema; // use dynamic vector to store the extrema
    for (size_t i = 1; i < values.size() - 1; i++)
    {
        if ((max && values(i) > values(i - 1) && values(i) > values(i + 1)) ||
            (!max && values(i) < values(i - 1) && values(i) < values(i + 1)))
            extrema.push_back(i);
    }

    // convert to Eigen vector
    const size_t numExtrema = extrema.size();
    Eigen::VectorXi eExtrema(numExtrema);
    for (size_t i = 0; i < numExtrema; i++)
        eExtrema(i) = extrema[i];
    return eExtrema;
}

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
        if (fabs(zScore) > criticalValue)
            return false;
    }
    // if (cumulativeZScore/3 > criticalValue)
    //     return false;
    return true;
}

} // namespace ORB_SLAM2