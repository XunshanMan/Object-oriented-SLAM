#pragma  once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "include/utils/matrix_utils.h"

#include "Ellipsoid.h"

namespace g2o
{
class plane {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    plane();

    plane(Vector4d param, Vector3d color=Vector3d(1.0,0.0,0.0));

    // Copy constructor.
    plane(const plane &p);

    const plane& operator=(const plane& p);

    // initialize a plane from a point and a normal vector
    void fromPointAndNormal(const Vector3d &point, const Vector3d &normal);

    void fromDisAndAngle(double dis, double angle);
    void fromDisAngleTrans(double dis, double angle, double trans);

    // update plane parameters in 3 degrees
    plane exp_update(const Vector3d& update);
    
    // update plane parameters in 2 degrees
    plane exp_update2DOF(const Vector2d& update);

    // distance from a point to plane
    double distanceToPoint(const Vector3d& point, bool keep_flag = false);
    void transform(g2o::SE3Quat& Twc);

    // init visualization for a finite plane given a point and a size
    void InitFinitePlane(const Vector3d& center, double size);

    // update function for optimization
    inline void oplus(const Vector3d& v){
      //construct a normal from azimuth and evelation;
      double _azimuth=v[0];
      double _elevation=v[1];
      double s=std::sin(_elevation), c=std::cos(_elevation);
      Vector3d n (c*std::cos(_azimuth), c*std::sin(_azimuth), s) ;
      
      // rotate the normal
      Matrix3d R=rotation(normal());
      double d=distance()+v[2];         // why is plus?
      param.head<3>() = R*n;
      param(3) = -d;
      normalize(param);
    }

    // update function for optimization
    inline void oplus_dual(const Vector3d& v){
      //construct a normal from azimuth and evelation;
      double _azimuth=v[0];
      double _elevation=0;
      double s=std::sin(_elevation), c=std::cos(_elevation);
      Vector3d n (c*std::cos(_azimuth), c*std::sin(_azimuth), s) ;
      
      // rotate the normal
      Matrix3d R=rotation(normal());
      double d=distance()+v[1];         // why is plus?
      param.head<3>() = R*n;
      param(3) = -d;
      normalize(param);

      mdDualDis += v[2];
    }

    static inline void normalize(Vector4d& coeffs) {
      double n=coeffs.head<3>().norm();
      coeffs = coeffs * (1./n);
    }

    Vector3d normal() const {
      return param.head<3>();
    }

    static Matrix3d rotation(const Vector3d& v)  {
      double _azimuth = azimuth(v);
      double _elevation = elevation(v); 
      return (AngleAxisd(_azimuth,  Vector3d::UnitZ())* AngleAxisd(- _elevation, Vector3d::UnitY())).toRotationMatrix();
    }

    // self
    double azimuth() const {
      return atan2(param[1], param[0]);
    }

    static double azimuth(const Vector3d& v) {
    return std::atan2(v(1),v(0));
    }

    static  double elevation(const Vector3d& v) {
    return std::atan2(v(2), v.head<2>().norm());
    }

    double distance() const {
      return -param(3);
    }

    Eigen::Vector4d GeneratePlaneVec();
    Eigen::Vector4d GenerateAnotherPlaneVec();

    // finite plane parameters; treat as a square
    double mdPlaneSize; // side length (meter)
    bool mbLimited;

    // dual plane parameter
    double mdDualDis;

    Vector4d param; // A B C : AX+BY+CZ+D=0
    Vector3d color; // r g b , [0,1.0]
    Vector3d mvPlaneCenter; // the center of the square. roughly defined.    
private:

    Eigen::Vector3d GetLineFromCenterAngle(const Eigen::Vector2d center, double angle);
    Eigen::Vector4d LineToPlane(const Eigen::Vector3d line);

};


} // namespace g2o

