// Single-frame ellipsoid extraction from RGB-D data

#ifndef ELLIPSOIDSLAM_ELLIPSOIDEXTRACTOR_H
#define ELLIPSOIDSLAM_ELLIPSOIDEXTRACTOR_H

#include <opencv2/opencv.hpp>

#include <core/Ellipsoid.h>
#include <core/Geometry.h>
#include <core/Map.h>
#include <core/BasicEllipsoidEdges.h>

#include <src/symmetry/PointCloudFilter.h>
#include <src/symmetry/Symmetry.h>
#include <src/symmetry/SymmetrySolver.h>

#include <Eigen/Core>

#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>		  

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

namespace EllipsoidSLAM
{

struct PCAResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    bool result;
    Eigen::Vector3d center;
    Eigen::Matrix3d rotMat;
    Eigen::Vector3d covariance;
    Eigen::Vector3d scale;
};

class EllipsoidExtractor
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EllipsoidExtractor();   

    // open symmetry plane estimation to finish point cloud completion
    void OpenSymmetry();

    // The supporting plane is used to segment object points and estimate the object orientation
    void SetSupportingPlane(g2o::plane* pPlane);
  
    // API: estimate a 3d ellipsoid from RGB-D data and a bounding box, the ellipsoid is in global coordinate
    g2o::ellipsoid EstimateLocalEllipsoid(cv::Mat& depth, Eigen::Vector4d& bbox, int label, Eigen::VectorXd &pose, camera_intrinsic& camera);   

    void OpenVisualization(Map* pMap);   // if opened, the pointcloud during the process will be visualized 
    void ClearPointCloudList(); // clear the visualized point cloud 


    bool GetResult();   // get extraction result.
    SymmetryOutputData GetSymmetryOutputData();  // get the detail of symmetry estimation
    EllipsoidSLAM::PointCloud* GetPointCloudInProcess();   // get the object point cloud
    EllipsoidSLAM::PointCloud* GetPointCloudDebug();   // get the debug point cloud before Eucliden filter

private:
    void LoadSymmetryPrior();  // define object symmetry prior

    pcl::PointCloud<PointType>::Ptr ExtractPointCloud(cv::Mat& depth, Eigen::Vector4d& bbox, Eigen::VectorXd &pose, camera_intrinsic& camera);    // extract point cloud from depth image.
    PCAResult ProcessPCA(pcl::PointCloud<PointType>::Ptr &pCloudPCL);   // apply principal component analysis 
    g2o::ellipsoid ConstructEllipsoid(PCAResult &data);   // generate a sparse ellipsoid estimation from PCA result.

    void ApplyGravityPrior(PCAResult &data);    // add the supporting groundplane prior to calibrate the rotation matrix

    EllipsoidSLAM::PointCloud* ApplyEuclideanFilter(EllipsoidSLAM::PointCloud* pCloud, Vector3d &center);   // apply euclidean filter to get the object points
    EllipsoidSLAM::PointCloud* ApplyPlaneFilter(EllipsoidSLAM::PointCloud* pCloud, double z);    // filter points lying under the supporting plane

    EllipsoidSLAM::PointCloud* ApplySupportingPlaneFilter(EllipsoidSLAM::PointCloud* pCloud);   

    bool GetCenter(cv::Mat& depth, Eigen::Vector4d& bbox, Eigen::VectorXd &pose, camera_intrinsic& camera, Vector3d& center); // get a coarse 3d center of the object
    double getDistanceFromPointToCloud(Vector3d& point, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud);  // get the minimum distance between a point and a pointcloud

    // make sure the rotation matrix is right-handed
    void AdjustChirality(PCAResult &data);

    // adjust the axis order to make z axis near the normal of the ground plane
    void AlignZAxisToGravity(PCAResult &data);
    // this function will be called if groundplane is set when aligning Z axis
    Eigen::Matrix3d calibRotMatAccordingToGroundPlane( Matrix3d& rotMat, const Vector3d& normal);

    void VisualizePointCloud(const string& name, EllipsoidSLAM::PointCloud* pCloud, const Vector3d &color = Vector3d(-1,-1,-1), int point_size = 2);
    void VisualizeEllipsoid(const string& name, g2o::ellipsoid* pObj);

    PCAResult ProcessPCANormalized(EllipsoidSLAM::PointCloud* pObject);

private:
    bool mResult;  // estimation result.

    int miEuclideanFilterState; // Euclidean filter result:  1 no clusters 2: not the biggest cluster 3: fail to find valid cluster 0: success
    int miSystemState;  // 0: normal 1: no depth value for center point 2: fail to filter. 3: no point left after downsample
    EllipsoidSLAM::PointCloud* mpPoints;    // store object points
    EllipsoidSLAM::PointCloud* mpPointsDebug;    // store points for debugging ( points before Euclidean filter)

    // supporting plane
    bool mbSetPlane;    
    g2o::plane* mpPlane;

    // symmetry prior
    std::map<int,int> mmLabelSymmetry;
    
public: 
    // data generated in the process
    Vector3d mDebugCenter;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> mDebugEuclideanFilterClouds;
    PointCloudPCL::Ptr mDebugCenterCloud;

private:
    SymmetrySolver* mpSymSolver;

    SymmetryOutputData mSymmetryOutputData;

    bool mbOpenVisualization;
    Map* mpMap;
    int miExtractCount;

    bool mbOpenSymmetry;
};


}

#endif // ELLIPSOIDSLAM_ELLIPSOIDEXTRACTOR_H