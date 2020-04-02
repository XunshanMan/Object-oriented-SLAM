#include "include/core/Ellipsoid.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

namespace g2o
{

class VertexEllipsoid:public BaseVertex<9,ellipsoid> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexEllipsoid(){};

    virtual void setToOriginImpl();

    virtual void oplusImpl(const double* update_);

    virtual bool read(std::istream& is) ;

    virtual bool write(std::ostream& os) const ;
};

// 6 degrees
class VertexEllipsoidXYZABC:public BaseVertex<6,ellipsoid> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexEllipsoidXYZABC(){};

    virtual void setToOriginImpl();

    virtual void oplusImpl(const double* update_);

    virtual bool read(std::istream& is) ;

    virtual bool write(std::ostream& os) const ;
};

// camera-object 3D error
class EdgeSE3Ellipsoid9DOF:public BaseBinaryEdge<9,ellipsoid, VertexSE3Expmap, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3Ellipsoid9DOF(){};

    virtual bool read(std::istream& is);

    virtual bool write(std::ostream& os) const;

    void computeError();
};

// camera -object 2D projection error, rectangle difference of two vertices (left-top and right-bottom points)
class EdgeSE3EllipsoidProj:public BaseBinaryEdge<4,Vector4d, VertexSE3Expmap, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3EllipsoidProj(){};

    virtual bool read(std::istream& is);

    virtual bool write(std::ostream& os) const;

    void computeError();

    Vector4d getProject();

    void setKalib(Matrix3d& K);

    Matrix3d Kalib;
};

// object supporting prior: the Z axis of the object must align with the normal direction of the supporting plane
class EdgeEllipsoidGravityPlanePrior:public BaseUnaryEdge<1,Vector4d, VertexEllipsoid>
{
public:
    virtual bool read(std::istream& is);

    virtual bool write(std::ostream& os) const;

    void computeError();
};

}