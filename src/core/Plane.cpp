#include "include/core/Plane.h"

namespace g2o
{
    plane::plane() {
        param = Vector4d(1,0,0,0);
        color = Vector3d(1,0,0);
        mdDualDis = 0;

        mbLimited = false;
    }

    plane::plane(Vector4d param_, Eigen::Vector3d color_) {
        param = param_;
        color = color_;
        mdDualDis = 0;

        mbLimited = false;
    }

    plane plane::exp_update(const Vector3d& update)
    {
        g2o::plane plane_update;

        plane_update.param[0] = param[0] + update[0]; // A
        plane_update.param[1] = param[1] + update[1]; // B
        plane_update.param[2] = 0; // C 
        plane_update.param[3] = param[3] + update[2]; // D
        plane_update.color = color;

        return plane_update;
    }

    plane plane::exp_update2DOF(const Vector2d& update)
    {
        double k_update = update[0];
        double b_update = update[1];

        double k_current = -param[0]/param[1];
        double b_current = -param[3]/param[1];

        double k = k_current + k_update;
        double b = b_current + b_update;
    
        double B = -1;
        double A = k;
        double C = 0;
        double D = b;

        g2o::plane plane_update;
        plane_update.param[0] = A; // A
        plane_update.param[1] = B; // B
        plane_update.param[2] = C; // C ; z remains unchanged
        plane_update.param[3] = D; // D
        plane_update.color = color;

        return plane_update;
    }

    plane::plane(const plane &p){
        param = p.param;
        color = p.color;

        mbLimited = p.mbLimited;
        mvPlaneCenter = p.mvPlaneCenter;
        mdPlaneSize = p.mdPlaneSize;

        mdDualDis = p.mdDualDis;
    }

    const plane& plane::operator=(const plane& p){
        param = p.param;
        color = p.color;

        mbLimited = p.mbLimited;
        mvPlaneCenter = p.mvPlaneCenter;
        mdPlaneSize = p.mdPlaneSize;

        mdDualDis = p.mdDualDis;
        return p;
    }

    void plane::fromPointAndNormal(const Vector3d &point, const Vector3d &normal)
    {
        param.head(3) = normal;  // normal : [ a, b, c]^T
        param[3] = -point.transpose() * normal;        // X^T * pi = 0 ; ax0+by0+cz0+d=0
        color = Vector3d(1,0,0);

        mdDualDis = 0;
    }

    void plane::fromDisAndAngle(double dis, double angle)
    {
        fromDisAngleTrans(dis, angle, 0);
    }

    void plane::fromDisAngleTrans(double dis, double angle, double trans)
    {
        param[0] = sin(angle);
        param[1] = -cos(angle);
        param[2] = 0;
        param[3] = -dis;

        mdDualDis = trans;
    }


    double plane::distanceToPoint(const Vector3d& point, bool keep_flag){
        double fenzi = param(0)*point(0) + param(1)*point(1) +param(2)*point(2) + param(3);
        double fenmu = std::sqrt ( param(0)*param(0)+param(1)*param(1)+param(2)*param(2) );
        double value = fenzi/fenmu;

        if(keep_flag) return value;
        else return std::abs(value);
    }

    void plane::transform(g2o::SE3Quat& Twc){
        Matrix4d matTwc = Twc.to_homogeneous_matrix();
        Matrix4d matTwc_trans = matTwc.transpose();
        Matrix4d matTwc_trans_inv = matTwc_trans.inverse();
        param = matTwc_trans_inv * param;
    }

    // for the visualization of symmetry planes
    void plane::InitFinitePlane(const Vector3d& center, double size)
    {
        mdPlaneSize = size;
        mvPlaneCenter = center;
        mbLimited = true;
    }

    Eigen::Vector4d plane::GeneratePlaneVec()
    {
        return param;
    }

    Eigen::Vector4d plane::GenerateAnotherPlaneVec()
    {
        // azimuth : angle of the normal
        g2o::plane p2;
        p2.fromDisAndAngle(mdDualDis, azimuth());

        return p2.param;
    }

    Vector3d plane::GetLineFromCenterAngle(const Vector2d center, double angle)
    {
        // x = center[0] + t * cos(theta)
        // y = center[1] + t * sin(theta)
        // goal : 
        // AX + BY + C = 0 ;  get A,B,C

        // get rid of t:
        // sin(theta) * x - cos(theta) * y = 
        //                      sin(theta) * center[0] - cos(theta) * center[1]
        
        // so: 
        // sint * x + (- cost) * y  + (cost*c1 - sint*c0) = 0

        Vector3d param;
        param[0] = sin(angle);
        param[1] = -cos(angle);
        param[2] = cos(angle) * center[1] - sin(angle) * center[0];

        return param;
    }

    Vector4d plane::LineToPlane(const Vector3d line)
    {
        Vector4d plane;
        plane << line[0], line[1], 0, line[2];
        return plane;
    }

} // g2o namespace