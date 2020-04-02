#include "core/Geometry.h"
#include "utils/matrix_utils.h"
#include "utils/dataprocess_utils.h"
namespace EllipsoidSLAM
{

    // Generates all points in the depth image as point cloud.
    PointCloud getPointCloud(cv::Mat &depth, cv::Mat &rgb, Eigen::VectorXd &detect, EllipsoidSLAM::camera_intrinsic &camera) {
        PointCloud cloud;

        int x1 = 0;
        int y1 = 0;
        int x2 = depth.cols;
        int y2 = depth.rows;

        for (int y = y1; y < y2; y++){
            for (int x = x1; x < x2; x++) {
                ushort *ptd = depth.ptr<ushort>(y);
                ushort d = ptd[x];

                PointXYZRGB p;
                p.z = d / camera.scale;
                if (p.z <= 0.1 || p.z > 100)           // limit the depth range in [0.1,100]m
                    continue;

                p.x = (x - camera.cx) * p.z / camera.fx;
                p.y = (y - camera.cy) * p.z / camera.fy;

                // get rgb color
                p.b = rgb.ptr<uchar>(y)[x * 3];
                p.g = rgb.ptr<uchar>(y)[x * 3 + 1];
                p.r = rgb.ptr<uchar>(y)[x * 3 + 2];

                p.size = 1;

                cloud.push_back(p);
            }
        }

        return cloud;
    }

    // apply transformation on every points in the pointcloud
    PointCloud* transformPointCloud(PointCloud *pPoints_local, g2o::SE3Quat* pCampose_wc)
    {
        Eigen::Matrix4d Twc = pCampose_wc->to_homogeneous_matrix();

        auto * pPoints_global = new PointCloud();
        for (auto &p : *pPoints_local) {
            Eigen::Vector3d xyz(p.x, p.y, p.z);
            Eigen::Vector3d xyz_w = TransformPoint(xyz, Twc);
            
            PointXYZRGB p_w;
            p_w.x = xyz_w[0];
            p_w.y = xyz_w[1];
            p_w.z = xyz_w[2];
            p_w.r = p.r;
            p_w.g = p.g;
            p_w.b = p.b;
            p_w.size = p.size;
            
            pPoints_global->push_back(p_w);
        }

        return pPoints_global;
    }

    void transformPointCloudSelf(PointCloud *pPoints_local, g2o::SE3Quat* pCampose_wc)
    {
        Eigen::Matrix4d Twc = pCampose_wc->to_homogeneous_matrix();

        for (auto &p : *pPoints_local) {
            Eigen::Vector3d xyz(p.x, p.y, p.z);
            Eigen::Vector3d xyz_w = TransformPoint(xyz, Twc);
            
            p.x = xyz_w[0];
            p.y = xyz_w[1];
            p.z = xyz_w[2];
        }

        return;
    }

    PointCloud loadPointsToPointVector(Eigen::MatrixXd &pMat){
        int total_num = pMat.rows();
        PointCloud points;

        for( int i=0; i<total_num; i++)
        {
            Eigen::VectorXd pVector = pMat.row(i);  // id x y z
            PointXYZRGB p;
            p.x = pVector(0);
            p.y = pVector(1);
            p.z = pVector(2);

            p.r = 0;
            p.g = 255;  // set green by default
            p.b = 0;

            p.size = 10;

            points.push_back(p);
        }
        return points;
    }

    void SetPointCloudProperty(PointCloud* pCloud, uchar r, uchar g, uchar b, int size){
        if(pCloud == NULL) return;
        for(PointXYZRGB& p : *pCloud)
        {
            p.r = r;
            p.g = g;
            p.b = b;
            p.size = size;
        }
        return;
    }

    Matrix3d getCalibFromCamera(camera_intrinsic &camera)
    {
        Matrix3d calib;
        calib<<camera.fx,  0,  camera.cx,  
        0,  camera.fy, camera.cy,
        0,      0,     1;
        return calib;
    }


    void SavePointCloudToTxt(const string& path, PointCloud* pCloud){
        int num = pCloud->size();
        MatrixXd outMat; outMat.resize(num, 3);
        for( int i=0;i<num;i++)
        {
            auto p = (*pCloud)[i];
            VectorXd pVec; pVec.resize(3);
            pVec << p.x,p.y,p.z;
            outMat.row(i) = pVec.transpose();
        }

        saveMatToFile(outMat, path.c_str());
    }

    Vector3d GetPointcloudCenter(PointCloud* pCloud)
    {
        Vector3d center;
        if(pCloud == NULL) return center;
        int num = pCloud->size();
        Vector3d total(0,0,0);
        for( int i=0; i<num; i++)
        {
            auto p = (*pCloud)[i];
            total[0] += p.x;
            total[1] += p.y;
            total[2] += p.z;
        }
        center = total / double(num);
        return center;
    }

    Vector3d TransformPoint(Vector3d &point, const Eigen::Matrix4d &T)
    {
        Eigen::Vector4d Xc = real_to_homo_coord<double>(point);
        Eigen::Vector4d Xw = T * Xc;
        Eigen::Vector3d xyz_w = homo_to_real_coord<double>(Xw);
        return xyz_w;
    }


}
