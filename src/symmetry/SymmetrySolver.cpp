#include "core/Ellipsoid.h"
#include "SymmetrySolver.h"
#include "PointCloudFilter.h"
#include "utils/matrix_utils.h"
// g2o
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

#include "Thirdparty/g2o/g2o/core/robust_kernel.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"

#include <src/config/Config.h>
#include <opencv2/opencv.hpp> 

using namespace EllipsoidSLAM;

static int calcu_time = 0;

bool isInRange(int x, int y, int range_x, int range_x2, int range_y, int range_y2)
{
    if( range_x < x && x < range_x2 )
        if( range_y<y && y< range_y2 )
            return true;
    return false;
}

bool isInFrustrum(int x, int y, int range_x, int range_y)
{
    return isInRange(x,y,0,range_x,0,range_y);
}

bool checkValidPoint(EllipsoidSLAM::PointXYZRGB &p)
{
    return (pcl_isfinite(p.x) && pcl_isfinite(p.y) && pcl_isfinite(p.z));
}

SymmetrySolver::SymmetrySolver(){
    mbOpenSparseEstimation = false;
}

// The rule of calculating probability:
// A mirrored pointcloud is generated from the original pointcloud using a hypothetical symmetry plane/ or dual plane.
// For every point in the mirrored point cloud, there are three possible situations:
// 1) it lies in the occluded area, the cost is set to zero
// 2) it lies in the observable area, the cost is set as the distance to the nearest original point.
double SymmetrySolver::GetPointCloudProb(Vector4d &bbox, PointCloud* pCloudSym, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree)
{
    g2o::ellipsoid e;
    g2o::SE3Quat campose_wc, campose_cw; 
    campose_wc.fromVector(pose.tail(7));
    campose_cw = campose_wc.inverse();
    Matrix3Xd projMat = e.generateProjectionMatrix(campose_cw, calib);   
    
    int num = pCloudSym->size();

    double ln_total_P = 0;

    int num_invalid = 0;
    for( int i=0; i<num; i++)
    {
        auto p = (*pCloudSym)[i];
        Vector4d pointHomo(p.x,p.y,p.z,1);

        Vector3d uvHomo = projMat * pointHomo;
        Vector2d uv = homo_to_real_coord_vec<double>(uvHomo);
        
        int x = int(uv[0]);
        int y = int(uv[1]);

        double dis_diff = 0;

        if( isInRange(x,y,bbox[0],bbox[2],bbox[1],bbox[3]) )   
        {
            ushort *ptd = depth.ptr<ushort>(int(uv[1]));   
            ushort d = ptd[int(uv[0])];

            if( d == 0 )    // occluded area
            {
                dis_diff = 0;
            }   
            else   
            {
                double depth = d / scale;

                Vector3d cam_c = campose_wc.toVector().head(3);
                Vector3d point(p.x,p.y,p.z);
                double dis_cam_point = (cam_c-point).norm();

                if( dis_cam_point > depth )  // occluded area
                {
                    dis_diff = 0;
                }
                else    // observable area
                {
                    if(!checkValidPoint(p)) {
                        dis_diff = 0;
                        num_invalid++;
                    }
                    else
                        dis_diff = findMinimalDistanceWithKdTree(p, kdTree);
                }
            }

        }
        else    // observable area
        {
            if(!checkValidPoint(p)) {
                dis_diff = 0;
                num_invalid++;
            }
            else
                dis_diff = findMinimalDistanceWithKdTree(p, kdTree);
        }

        // Sigma is used for probability calculation. it's useless in the optimization process.
        double Sigma = Config::ReadValue<double>("SymmetrySolver.Sigma");  
        double Sigma_inv = 1.0 / Sigma; 

        double ln_P = -0.5*Sigma_inv*Sigma_inv*dis_diff*dis_diff;
        ln_total_P += ln_P;        
    }

    int num_valid = num - num_invalid;
    double aver_ln_P;
    if(num_valid > 0)
        aver_ln_P = ln_total_P / double(num_valid);  
    else
        aver_ln_P = -INFINITY;
    mData.pSymetryCloud = pCloudSym;

    if(num_invalid>0)
        std::cout << " - Invalid/Valid SymPoints Num : " << num_invalid << " / " << num_valid << std::endl;

    return aver_ln_P;
}

double SymmetrySolver::findMinimalDistanceWithKdTree(PointXYZRGB &p, pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree)
{
    pcl::PointXYZ searchPoint;

    searchPoint.x = p.x;
    searchPoint.y = p.y;
    searchPoint.z = p.z;

    // K nearest neighbor search
    int K = 1;

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    kdTree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);
    
    assert( pointIdxNKNSearch.size() == 1 && "Error in KdTree Search." );

    int nearest_id = pointIdxNKNSearch[0];
    pcl::PointXYZ nearest_point = kdTree.getInputCloud()->points[nearest_id];

    Vector3d pos1(p.x,p.y,p.z);
    Vector3d pos2(nearest_point.x,nearest_point.y,nearest_point.z);
    double dis = (pos1-pos2).norm();

    return dis;
}

PointCloud* SymmetrySolver::GetSymmetryPointCloud(PointCloud* pCloud, g2o::plane &plane, int symType)
{
    PointCloud* pCloudSym = new PointCloud;

    int num = pCloud->size();
    for(int i=0;i<num;i++)
    {
        auto p = (*pCloud)[i];
        Vector3d point(p.x, p.y, p.z);
        Vector4d point_homo = real_to_homo_coord_vec<double>(point);
        Vector4d pointSymHomo = GetSymmetryPointOfPlane(point_homo, plane);
        Vector3d pointSym = homo_to_real_coord_vec<double>(pointSymHomo);
        
        auto pSym = p;

        pSym.g = 255;
        pSym.x = pointSym[0];
        pSym.y = pointSym[1];
        pSym.z = pointSym[2];
        pCloudSym->push_back(pSym);

    }
    return pCloudSym;
}

Vector4d SymmetrySolver::GetSymmetryPointOfPlane(Vector4d& point, g2o::plane &plane)
{
    // get the plane normal vector
    Vector3d normal = plane.param.head(3);
    normal.normalize();

    double up_equation = std::abs(plane.param.transpose()*point);
    double down_equation = std::sqrt(plane.param.head(3).transpose()*plane.param.head(3));
    double dis = up_equation/down_equation;

    Vector3d point_3 = homo_to_real_coord_vec<double>(point);


    double symbol;
    double symbol_value = point.transpose()*plane.param;
    if(symbol_value > 0) symbol =-1 ;
    else symbol = 1;

    Vector3d symPoint = point_3 + 2*symbol*dis*normal;
    Vector4d symPointHomo = real_to_homo_coord_vec<double>(symPoint);
    return symPointHomo;
}

// Given an initial symmetry plane, output an optimized symmetry plane
SymmetrySolverData SymmetrySolver::OptimizeSymmetryPlane(Vector4d &bbox, g2o::plane& initPlane, PointCloud* pCloud, cv::Mat &depth_origin, VectorXd &pose, Matrix3d &calib, double scale, 
        int symType)
{
    g2o::SparseOptimizer graph;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    graph.setAlgorithm(solver);
    graph.setVerbose(false);        // output

    // add vertices
    g2o::plane plane = initPlane;
    g2o::VertexPlane* vpVertexPlane = new g2o::VertexPlane;
    vpVertexPlane->setEstimate(plane);
    vpVertexPlane->setId(0);
    vpVertexPlane->setFixed(false);
    graph.addVertex(vpVertexPlane);

    // add edges
    g2o::EdgeSymmetryPlane* e = new g2o::EdgeSymmetryPlane();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vpVertexPlane ));
    e->setMeasurement(1); 
    e->setId(0);
    double inv_sigma;inv_sigma = 1;  
    inv_sigma = inv_sigma*1;        
    Matrix<double,1,1> info; info(0,0) = inv_sigma;
    e->setInformation(info);

    // initialize edge parameter
    e->initializeParam(bbox, pCloud, depth_origin, pose, calib, scale);
    e->initializeKDTree();    

    if( mbOpenSparseEstimation )
        e->setBordersPointCloud( mpBorders );

    graph.addEdge(e);

    // save 
    mData.pInitPlane = new g2o::plane(vpVertexPlane->estimate());

    e->computeError();
    Matrix<double, 1, 1> errorMat_init = e->error();
    mData.init_error = errorMat_init(0,0);

    graph.initializeOptimization();
    graph.optimize(5);      // the number could be adjusted according to your time-efficiency trade

    Matrix<double, 1, 1> errorMat = e->error();

    // save result.
    mData.prob = exp(-errorMat(0,0));   // probability calculation could be adjusted here.
    mData.pPlane = new g2o::plane(vpVertexPlane->estimate());
    mData.pPlane->color = Vector3d(1,0,0);
    mData.symType = symType;
    mData.final_error = errorMat(0,0);
    mData.result = true;
    return mData;
    
}

// this one is the dual plane version.
SymmetrySolverData SymmetrySolver::OptimizeSymmetryDualPlane(Vector4d &bbox, g2o::plane& initPlane, PointCloud* pCloud, cv::Mat &depth_origin, VectorXd &pose, Matrix3d &calib, double scale, 
        int symType)
{
    g2o::SparseOptimizer graph;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    graph.setAlgorithm(solver);
    graph.setVerbose(false);    
    
    g2o::VertexDualPlane* vpVertexDualPlane = new g2o::VertexDualPlane;
    vpVertexDualPlane->setEstimate(initPlane);
    vpVertexDualPlane->setId(0);
    vpVertexDualPlane->setFixed(false);
    graph.addVertex(vpVertexDualPlane);

    g2o::EdgeSymmetryDualPlane* e = new g2o::EdgeSymmetryDualPlane();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vpVertexDualPlane ));
    e->setMeasurement(1); 
    e->setId(0);
    double inv_sigma;inv_sigma = 1;  
    Matrix<double,1,1> info; info(0,0) = inv_sigma;
    e->setInformation(info);

    e->initializeParam(bbox, pCloud, depth_origin, pose, calib, scale);
    e->initializeKDTree();    

    if( mbOpenSparseEstimation )
        e->setBordersPointCloud( mpBorders );

    graph.addEdge(e);

    mData.pInitPlane = new g2o::plane(initPlane.GeneratePlaneVec());
    mData.pInitPlane2 = new g2o::plane(initPlane.GenerateAnotherPlaneVec());
    e->computeError();
    Matrix<double, 1, 1> errorMat_init = e->error();
    mData.init_error = errorMat_init(0,0);

    graph.initializeOptimization();
    graph.optimize(5); 

    Matrix<double, 1, 1> errorMat = e->error();

    mData.prob = exp(-errorMat(0,0));
    g2o::plane dualplaneOptimized = vpVertexDualPlane->estimate();
    mData.pPlane = new g2o::plane(dualplaneOptimized.GeneratePlaneVec());
    mData.pPlane2 = new g2o::plane(dualplaneOptimized.GenerateAnotherPlaneVec());
    mData.pPlane->color = Vector3d(1.0,0.3,0);
    mData.pPlane2->color = Vector3d(1.0,0.3,0);
    mData.symType = symType;
    mData.result = true;
    mData.final_error = errorMat(0,0);
    return mData;
}

void SymmetrySolver::SetCameraParam(camera_intrinsic& camera){
    mCamera = camera;
}

void SymmetrySolver::SetInitPlane(g2o::plane* plane){
    mpInitPlane = plane;
}

SymmetrySolverData SymmetrySolver::getResult()
{
    return mData;
}

void SymmetrySolver::SetBorders(EllipsoidSLAM::PointCloud* pBorders){
    mpBorders = pBorders;
    mbOpenSparseEstimation = true;
}

SymmetrySolverData SymmetrySolver::mData;

namespace g2o
{
    EdgeSymmetryPlane::EdgeSymmetryPlane()
    {
        mbInitilized = false;
        mbKdTreeInitialized = false;

        mpBorders = NULL;   
        miSymmetryType = 1; // default: reflection symmetry
    }

    bool EdgeSymmetryPlane::read(std::istream& is){
        return true;
    };

    bool EdgeSymmetryPlane::write(std::ostream& os) const
    {
        return os.good();
    };

    void EdgeSymmetryPlane::computeError()
    {
        if(!mbInitilized)
            cerr << "Symmetry Edge has not been initialized yet." << endl;
        const VertexPlane* planeVertex = static_cast<const VertexPlane*>(_vertices[0]);

        assert( mpCloud != NULL && "Point cloud shouldn't be NULL.");

        g2o::plane plane = planeVertex->estimate();

        // generate mirrored pointcloud
        PointCloud* pObjectCloud;
        if(mpBorders == NULL ) pObjectCloud = mpCloud;
        else pObjectCloud = mpBorders;
        PointCloud* pCloudSym = SymmetrySolver::GetSymmetryPointCloud(pObjectCloud, plane);

        double cost;
        assert( mbKdTreeInitialized == true && "Please initialize kdTree.");
        cost = SymmetrySolver::GetPointCloudProb(mBbox, pCloudSym, mpCloud, mDepth, mPose, mCalib, mScale, mKdtree);
        
        _error = Matrix<double, 1, 1>( - cost);
    }

    void EdgeSymmetryPlane::initializeParam(Vector4d &bbox, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale)
    {
        mpCloud = pCloud;
        mDepth = depth;
        mPose = pose;
        mCalib = calib;
        mScale = scale;

        mBbox = bbox;

        mbInitilized = true;
    }

    void EdgeSymmetryPlane::initializeKDTree()
    {
        if(mpCloud == NULL )
        {
            cerr << "Point cloud is NULL." << endl;
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCL (new pcl::PointCloud<pcl::PointXYZ>);    
        int num = mpCloud->size();
        for(int i=0;i<num;i++){
            EllipsoidSLAM::PointXYZRGB pT = (*mpCloud)[i];
            
            pcl::PointXYZ p;
            p.x = pT.x;
            p.y = pT.y;
            p.z = pT.z;
            cloudPCL->points.push_back(p);
        }

        mKdtree.setInputCloud (cloudPCL);
        mbKdTreeInitialized = true;
    }

    void EdgeSymmetryPlane::setBordersPointCloud(PointCloud* pBorders)
    {
        mpBorders = pBorders;
    }

    EdgeSymmetryDualPlane::EdgeSymmetryDualPlane()
    {
        mbInitilized = false;
        mbKdTreeInitialized = false;

        mpBorders = NULL; 
        miSymmetryType = 1; // default: reflection symmetry
    }

    bool EdgeSymmetryDualPlane::read(std::istream& is){
        return true;
    };

    bool EdgeSymmetryDualPlane::write(std::ostream& os) const
    {
        return os.good();
    };

    void EdgeSymmetryDualPlane::computeError()
    {
        if(!mbInitilized)
            cerr << "Symmetry Edge has not been initialized yet." << endl;
        const VertexDualPlane* dualplaneVertex = static_cast<const VertexDualPlane*>(_vertices[0]);

        assert( mpCloud != NULL && "Point cloud shouldn't be NULL.");

        g2o::plane dualplane = dualplaneVertex->estimate();
        g2o::plane plane1(dualplane.GeneratePlaneVec());
        g2o::plane plane2(dualplane.GenerateAnotherPlaneVec());

        PointCloud* pObjectCloud;
        if(mpBorders == NULL ) pObjectCloud = mpCloud;
        else pObjectCloud = mpBorders;

        PointCloud* pCloudSym1 = SymmetrySolver::GetSymmetryPointCloud(pObjectCloud, plane1);
        PointCloud* pCloudSym2 = SymmetrySolver::GetSymmetryPointCloud(pObjectCloud, plane2);
        CombinePointCloud(pCloudSym1, pCloudSym2);
        assert( mbKdTreeInitialized && " Please initialize the kdtree " );
        double cost = SymmetrySolver::GetPointCloudProb(mBbox, pCloudSym1, mpCloud, mDepth, mPose, mCalib, mScale, mKdtree);
        
        _error = Matrix<double, 1, 1>( - cost);
    }

    void EdgeSymmetryDualPlane::initializeParam(Vector4d &bbox, PointCloud* pCloud, cv::Mat& depth, VectorXd &pose, Matrix3d &calib, double scale)
    {
        mpCloud = pCloud;
        mDepth = depth;
        mPose = pose;
        mCalib = calib;
        mScale = scale;

        mBbox = bbox;

        mbInitilized = true;
    }

    void EdgeSymmetryDualPlane::initializeKDTree()
    {
        if(mpCloud == NULL )
        {
            cerr << "Point cloud is NULL." << endl;
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCL (new pcl::PointCloud<pcl::PointXYZ>);    
        int num = mpCloud->size();
        for(int i=0;i<num;i++){
            EllipsoidSLAM::PointXYZRGB pT = (*mpCloud)[i];
            
            pcl::PointXYZ p;
            p.x = pT.x;
            p.y = pT.y;
            p.z = pT.z;
            cloudPCL->points.push_back(p);
        }

        mKdtree.setInputCloud (cloudPCL);
        mbKdTreeInitialized = true;
    }

    void EdgeSymmetryDualPlane::setBordersPointCloud(PointCloud* pBorders)
    {
        mpBorders = pBorders;
    }

    void VertexPlane::setToOriginImpl() { _estimate = g2o::plane(); }

    void VertexPlane::oplusImpl(const double* update_) {
        Eigen::Map<const Vector2d> update(update_);
        Vector3d update2DOF(update[0], 0, update[1]);       // yaw, 0, dis

        _estimate.oplus(update2DOF);  
    }

    bool VertexPlane::read(std::istream& is) {
        return true;
    }

    bool VertexPlane::write(std::ostream& os) const {
        return os.good();
    }

    void VertexDualPlane::setToOriginImpl() { _estimate = g2o::plane(); }

    void VertexDualPlane::oplusImpl(const double* update_) {
        Eigen::Map<const Vector3d> update(update_);
        _estimate.oplus_dual(update);
    }

    bool VertexDualPlane::read(std::istream& is) {
        return true;
    }

    bool VertexDualPlane::write(std::ostream& os) const {
        return os.good();
    }

} // namespace g2o
