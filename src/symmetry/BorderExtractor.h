// extract borders from point objects to speed up the symmetry plane estimation
#ifndef ELLIPSOIDSLAM_BORDEREXTRACTOR_H
#define ELLIPSOIDSLAM_BORDEREXTRACTOR_H

#include <iostream>

#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/console/parse.h>

#include "src/symmetry/PointCloudFilter.h"

#include <core/Geometry.h>

#include <ctime>

typedef pcl::PointXYZ PointType;

namespace EllipsoidSLAM
{

class BorderExtractor
{
public:

pcl::PointCloud<PointType>::Ptr ToPointTCloud(pcl::PointCloud<pcl::PointWithRange>::Ptr pCloud);
pcl::PointCloud<PointType>::Ptr CombineCloud(pcl::PointCloud<PointType>::Ptr& pCloud1, pcl::PointCloud<PointType>::Ptr& pCloud2, pcl::PointIndices& indices1);
pcl::PointCloud<PointType>::Ptr FilterBordersBasedOnPointCloud(pcl::PointCloud<pcl::PointWithRange>::Ptr& pBordersNoisy, pcl::PointCloud<PointType>::Ptr &point_cloud_ptr);
pcl::PointCloud<pcl::PointWithRange>::Ptr extractBordersFromPointCloud(pcl::PointCloud<PointType>::Ptr& point_cloud_ptr, double resolution);

};  // class

}// namespace

#endif  //ELLIPSOIDSLAM_BORDEREXTRACTOR_H