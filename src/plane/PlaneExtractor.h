// this class offers a simple demo for extracting the groundplane from a single RGB-D frame

#ifndef PLANEEXTRACTOR_H
#define PLANEEXTRACTOR_H

#include <iostream>
#include <string>

#include <core/Plane.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudPCL;
using namespace std;

namespace EllipsoidSLAM
{

struct PlaneExtractorParam
{
    double fx,fy,cx,cy;
    double scale;

    bool RangeOpen = false; // if this flag is opened, only the bottom half part of the depth image is considered
    int RangeHeight;
};

class PlaneExtractor
{

public:
    PlaneExtractor(){};
    PlaneExtractor(const string& settings);
    bool extractGroundPlane(const cv::Mat &depth, g2o::plane& plane);

    void extractPlanes(const cv::Mat &depth);

    void SetParam(PlaneExtractorParam& param);

    std::vector<PointCloudPCL> GetPoints();
    std::vector<PointCloudPCL> GetPotentialGroundPlanePoints();

    std::vector<cv::Mat> GetCoefficients();
    PointCloudPCL::Ptr GetCloudDense();
private:
    // params
    int mParamRangeHeight;  // whether to consider the bottom half part only
    
    PlaneExtractorParam mParam;

    std::vector<PointCloudPCL> mvPlanePoints;
    std::vector<PointCloudPCL> mvPotentialGroundPlanePoints;
    std::vector<cv::Mat> mvPlaneCoefficients;

    PointCloudPCL::Ptr mpCloudDense;
};


} // EllipsoidSLAM

#endif // PLANEEXTRACTOR_H