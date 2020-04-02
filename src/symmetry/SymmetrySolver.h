#ifndef ELLIPSOIDSLAM_SYMMETRYSOLVER_H
#define ELLIPSOIDSLAM_SYMMETRYSOLVER_H

#include <core/Geometry.h>
#include <core/Ellipsoid.h>
#include <core/Plane.h>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

using namespace Eigen;
using namespace EllipsoidSLAM;

class SymmetrySolverData
{
public:
    PointCloud* pSymetryCloud;
    g2o::plane* pPlane;
    g2o::plane* pPlane2; 

    double prob;
    int symType;
    bool result;
    
    // save the initial data
    g2o::plane* pInitPlane;
    g2o::plane* pInitPlane2;
    double init_error;
    double final_error;
};

class SymmetrySolver
{

public:
    SymmetrySolver();
    
    SymmetrySolverData OptimizeSymmetryPlane(Vector4d &bbox, g2o::plane& initPlane, PointCloud* pCloud, cv::Mat &depth_origin, VectorXd &pose, Matrix3d &calib, double scale, int symType);
    SymmetrySolverData OptimizeSymmetryDualPlane(Vector4d &bbox, g2o::plane& initPlane, PointCloud* pCloud, cv::Mat &depth_origin, VectorXd &pose, Matrix3d &calib, double scale, int symType);
    
    static double GetPlaneProbWithKdTree(g2o::plane &plane, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree, 
            PointCloud* pBorders = NULL);

    static double GetPointCloudProb(Vector4d &bbox, PointCloud* pCloudSym, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree);
    static double GetPlaneProbWithKdTreeUsingAdvanceProbEquation(g2o::plane &plane, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree, 
        PointCloud* pBorders = NULL);

    static double calculatePlaneCost(g2o::plane &plane,  PointCloud* pCloud);

    static PointCloud* GetSymmetryPointCloud(PointCloud* pCloud, g2o::plane &p, int symType = 1);

public:
    void SetBorders(EllipsoidSLAM::PointCloud* pBorders);

private:
    static double calculateOcclProbOfPoints(PointCloud *pCloudSym, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale);
    static double calculateMatchProbOfPoints(PointCloud* pCloud,  PointCloud* pCloudSym);
    static double findMinimalDistance(PointXYZRGB &p, PointCloud* pCloud);
    static Vector4d GetSymmetryPointOfPlane(Vector4d& point, g2o::plane &plane);

    static double calculateMatchProbOfPointsWithKdTree(PointCloud* pCloud,  PointCloud* pCloudSym, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree);
    static double findMinimalDistanceWithKdTree(PointXYZRGB &p, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree);

public:

    static SymmetrySolverData mData;

private:
    bool mbOpenSparseEstimation;
    PointCloud* mpBorders;

    std::map<int,int> mmLabelSymmetry;

public:
    void SetCameraParam(camera_intrinsic& camera);
    void SetInitPlane(g2o::plane* plane);
    SymmetrySolverData getResult();
private:
    camera_intrinsic mCamera;
    g2o::plane* mpInitPlane;

};

/*
*   This part contains Vertex, Edges in symmetry optimization using g2o.
*/
namespace g2o
{
    class VertexPlane:public BaseVertex<2,g2o::plane> 
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexPlane(){};

        virtual void setToOriginImpl();

        virtual void oplusImpl(const double* update_);

        virtual bool read(std::istream& is) ;

        virtual bool write(std::ostream& os) const ;

    };

    // Dual Plane 
    class VertexDualPlane:public BaseVertex<3,g2o::plane> 
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        VertexDualPlane(){};

        virtual void setToOriginImpl();

        virtual void oplusImpl(const double* update_);

        virtual bool read(std::istream& is) ;

        virtual bool write(std::ostream& os) const ;

    };

    class EdgeSymmetryPlane:public BaseUnaryEdge<1,double, VertexPlane>
    {
    public:
        EdgeSymmetryPlane();

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;

        void computeError();

        void initializeParam(Vector4d &bbox, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale);
        void initializeKDTree();

        void setBordersPointCloud(PointCloud* pBorders);
    private:
        EllipsoidSLAM::PointCloud* mpCloud;
        cv::Mat mDepth;
        VectorXd mPose;
        Matrix3d mCalib;
        double mScale;

        Vector4d mBbox; //bounding box.

        bool mbInitilized;

        pcl::KdTreeFLANN<pcl::PointXYZ> mKdtree;
        bool mbKdTreeInitialized;

        EllipsoidSLAM::PointCloud* mpBorders;

        int miSymmetryType;

    };


    class EdgeSymmetryDualPlane:public BaseUnaryEdge<1,double, VertexDualPlane>
    {
    public:
        EdgeSymmetryDualPlane();

        virtual bool read(std::istream& is);

        virtual bool write(std::ostream& os) const;

        void computeError();

        void initializeParam(Vector4d &bbox, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale);
        void initializeKDTree();

        void setBordersPointCloud(PointCloud* pBorders);

        void setSymmetryType(int type);

    private:
        EllipsoidSLAM::PointCloud* mpCloud;
        cv::Mat mDepth;
        VectorXd mPose;
        Matrix3d mCalib;
        double mScale;

        Vector4d mBbox; //bounding box.

        bool mbInitilized;

        pcl::KdTreeFLANN<pcl::PointXYZ> mKdtree;
        bool mbKdTreeInitialized;

        EllipsoidSLAM::PointCloud* mpBorders;

        int miSymmetryType;

    };
}


#endif