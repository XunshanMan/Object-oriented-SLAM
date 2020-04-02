#ifndef ELLIPSOIDSLAM_MAPDRAWER_H
#define ELLIPSOIDSLAM_MAPDRAWER_H

#include "Map.h"
#include<pangolin/pangolin.h>

#include<mutex>
#include <string>
#include<map>

using namespace std;

namespace EllipsoidSLAM{

class Map;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapDrawer(const string &strSettingPath, Map* pMap);

    bool updateObjects();
    bool updateCameraState();


    bool drawObjects();
    bool drawCameraState();
    bool drawEllipsoids();

    bool drawPlanes();

    bool drawPoints();

    void setCalib(Eigen::Matrix3d& calib);

    bool drawTrajectory();

    void SE3ToOpenGLCameraMatrix(g2o::SE3Quat &matIn, pangolin::OpenGlMatrix &M); // inverse matIn
    void SE3ToOpenGLCameraMatrixOrigin(g2o::SE3Quat &matIn, pangolin::OpenGlMatrix &M); // don't inverse matIn
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

    void drawPointCloudLists(); // draw all the point cloud lists 
    void drawPointCloudWithOptions(const std::map<std::string,bool> &options); // draw the point cloud lists with options opened

private:

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;

    Map* mpMap;

    Eigen::Matrix3d mCalib;  

    void drawPlaneWithEquation(plane* p);

    pangolin::OpenGlMatrix getGLMatrixFromCenterAndNormal(Vector3f& center, Vector3f& normal);
};
}

#endif //ELLIPSOIDSLAM_MAPDRAWER_H
