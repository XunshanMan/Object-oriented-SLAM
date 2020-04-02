#ifndef ELLIPSOIDSLAM_GEOMETRY_H
#define ELLIPSOIDSLAM_GEOMETRY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

using namespace std;
using namespace Eigen;

namespace EllipsoidSLAM
{
    struct PointXYZRGB
    {
        double x;
        double y;
        double z;
        unsigned char r;    // 0-255
        unsigned char g;
        unsigned char b;
        int size = 1;
    };
    typedef vector<PointXYZRGB> PointCloud;

    struct camera_intrinsic{
        double fx;
        double fy;
        double cx;
        double cy;
        double scale;
    };

    PointCloud getPointCloud(cv::Mat &depth, cv::Mat &rgb, VectorXd &detect, camera_intrinsic &camera);
    PointCloud* transformPointCloud(PointCloud *pPoints_local, g2o::SE3Quat* pCampose_wc);  // generate a new cloud
    void transformPointCloudSelf(PointCloud *pPoints_local, g2o::SE3Quat* pCampose_wc);  // change the points of the origin cloud

    PointCloud loadPointsToPointVector(Eigen::MatrixXd &pMat);

    void SetPointCloudProperty(PointCloud* pCloud, uchar r, uchar g, uchar b, int size);

    Matrix3d getCalibFromCamera(camera_intrinsic &camera);

    void SavePointCloudToTxt(const string& path, PointCloud* pCloud);

    Vector3d GetPointcloudCenter(PointCloud* pCloud);
    Vector3d TransformPoint(Vector3d &point, const Eigen::Matrix4d &T);
}

#endif //ELLIPSOIDSLAM_GEOMETRY_H
