// process point cloud: segmentation, downsample, filter...

#ifndef ELLIPSOIDSLAM_POINTCLOUDFILTER_H
#define ELLIPSOIDSLAM_POINTCLOUDFILTER_H

// pcl
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudPCL;

#include <core/Geometry.h>
#include <core/Ellipsoid.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#include <ctime>

Vector2d getXYCenterOfPointCloud(EllipsoidSLAM::PointCloud* pPoints);
EllipsoidSLAM::PointCloud getPointCloudInRect(cv::Mat &depth, cv::Mat &rgb, const VectorXd &detect, EllipsoidSLAM::camera_intrinsic &camera, double range=100);
EllipsoidSLAM::PointCloud getPointCloudInRect(cv::Mat &depth, const VectorXd &detect, EllipsoidSLAM::camera_intrinsic &camera, double range=100);
void filterGround(EllipsoidSLAM::PointCloud** ppCloud);
void outputCloud(EllipsoidSLAM::PointCloud *pCloud, int num = 10);
EllipsoidSLAM::PointCloud pclToQuadricPointCloud(PointCloudPCL::Ptr &pCloud);
PointCloudPCL::Ptr QuadricPointCloudToPcl(EllipsoidSLAM::PointCloud &cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr QuadricPointCloudToPclXYZ(EllipsoidSLAM::PointCloud &cloud);

EllipsoidSLAM::PointCloud* pclToQuadricPointCloudPtr(PointCloudPCL::Ptr &pCloud);

EllipsoidSLAM::PointCloud pclXYZToQuadricPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &pCloud);
EllipsoidSLAM::PointCloud* pclXYZToQuadricPointCloudPtr(pcl::PointCloud<pcl::PointXYZ>::Ptr &pCloud);

// downsample and outlier filter
void DownSamplePointCloud(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, int param_num = 100);

// only downsample
void DownSamplePointCloudOnly(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, double grid=0.02);

// filter outliers
void FiltOutliers(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, int num_neighbor = 100);

// filter points and keep those in the ellipsoid
void FiltPointsInBox(EllipsoidSLAM::PointCloud* pPoints_global, EllipsoidSLAM::PointCloud* pPoints_global_inBox, g2o::ellipsoid &e);

void CombinePointCloud(EllipsoidSLAM::PointCloud *p1, EllipsoidSLAM::PointCloud *p2);

#endif // POINTCLOUDFILTER