#pragma  once

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "include/utils/matrix_utils.h"
 
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 3, 8> Matrix38d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace g2o
{


class ellipsoid{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3Quat pose;  // rigid body transformation, object in world coordinate
    Vector3d scale; // a,b,c : half length of axis x,y,z

    Vector9d vec_minimal; // x,y,z,roll,pitch,yaw,a,b,c

    double prob;    // probability from single-frame ellipsoid estimation

    int miLabel;        // semantic label.
    int miInstanceID;   // instance id.

    ellipsoid();
    // Copy constructor.
    ellipsoid(const ellipsoid &e);
    const ellipsoid& operator=(const ellipsoid& in);

    // xyz roll pitch yaw half_scale
    void fromMinimalVector(const Vector9d& v);

    // xyz quaternion, half_scale
    void fromVector(const Vector10d& v);

    const Vector3d& translation() const;
    void setTranslation(const Vector3d& t_);
    void setRotation(const Quaterniond& r_);
    void setRotation(const Matrix3d& R);
    void setScale(const Vector3d &scale_);

    // apply update to current ellipsoid. exponential map
    ellipsoid exp_update(const Vector9d& update);
    ellipsoid exp_update_XYZABC(const Vector6d& update);

    // actual error between two ellipsoid.
    Vector9d ellipsoid_log_error_9dof(const ellipsoid& newone) const;

    // change front face by rotate along current body z axis. another way of representing cuboid. representing same cuboid (IOU always 1)
    ellipsoid rotate_ellipsoid(double yaw_angle) const;

    // function called by g2o.
    Vector9d min_log_error_9dof(const ellipsoid& newone, bool print_details=false) const;

    // transform a local cuboid to global cuboid  Twc is camera pose. from camera to world
    ellipsoid transform_from(const SE3Quat& Twc) const;

    // transform a global cuboid to local cuboid  Twc is camera pose. from camera to world
    ellipsoid transform_to(const SE3Quat& Twc) const;

    // xyz roll pitch yaw half_scale
    Vector9d toMinimalVector() const;

    // xyz quaternion, half_scale
    Vector10d toVector() const;

    Matrix4d similarityTransform() const;

    // Get the projected point of ellipsoid center on image plane
    Vector2d projectCenterIntoImagePoint(const SE3Quat& campose_cw, const Matrix3d& Kalib);

    // The ellipsoid is projected onto the image plane to get an ellipse
    Vector5d projectOntoImageEllipse(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;

    // Get projection matrix P = K [ R | t ]
    Matrix3Xd generateProjectionMatrix(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;

    // Get Q^*
    Matrix4d generateQuadric() const;

    // Get the bounding box from ellipse in image plane
    Vector4d getBoundingBoxFromEllipse(Vector5d &ellipse) const;

    // Get the projected bounding box in the image plane of the ellipsoid using a camera pose and a calibration matrix.
    Vector4d getBoundingBoxFromProjection(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;

    // *** The following 4 functions treat the ellipsoid as external cube
    // 3x8 matrix storing 8 corners; each row is x y z
    Matrix3Xd compute3D_BoxCorner() const;

    Matrix2Xd projectOntoImageBoxCorner(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;

    // get rectangles after projection  [topleft, bottomright]
    Vector4d projectOntoImageRect(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;

    // get rectangles after projection  [center, width, height]
    Vector4d projectOntoImageBbox(const SE3Quat& campose_cw, const Matrix3d& Kalib) const;


    // calculate the IoU error with another ellipsoid
    double calculateMIoU(const g2o::ellipsoid& e) const;
    double calculateIntersectionOnZ(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const;
    double calculateArea(const g2o::ellipsoid& e) const;
    double calculateIntersectionOnXY(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const;
    double calculateIntersectionError(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const;

    void setColor(const Vector3d &color, double alpha = 1.0);
    Vector3d getColor();
    Vector4d getColorWithAlpha();
    bool isColorSet();

    // whether the camera could see the ellipsoid
    bool CheckObservability(const SE3Quat& campose_cw);

private:
    bool mbColor;
    Vector4d mvColor;

    void UpdateValueFrom(const g2o::ellipsoid& e);      // update the basic parameters from the given ellipsoid
};

} // g2o
