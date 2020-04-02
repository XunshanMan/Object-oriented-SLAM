#ifndef ELLIPSOIDSLAM_MAP_H
#define ELLIPSOIDSLAM_MAP_H

#include "Ellipsoid.h"
#include "Geometry.h"
#include "Plane.h"
#include <mutex>
#include <set>

#include <opencv2/opencv.hpp>

using namespace g2o;

namespace EllipsoidSLAM
{
    class Map
    {
    public:
        Map();

        void addEllipsoid(ellipsoid* pObj);
        std::vector<ellipsoid*> GetAllEllipsoids();

        void addPlane(plane* pPlane);
        std::vector<plane*> GetAllPlanes();
        void clearPlanes();

        void setCameraState(g2o::SE3Quat* state);
        g2o::SE3Quat* getCameraState();

        void addCameraStateToTrajectory(g2o::SE3Quat* state);
        std::vector<g2o::SE3Quat*> getCameraStateTrajectory();

        void addPoint(PointXYZRGB* pPoint);
        void addPointCloud(PointCloud* pPointCloud);
        void clearPointCloud();
        std::vector<PointXYZRGB*> GetAllPoints();

        std::vector<ellipsoid*> getEllipsoidsUsingLabel(int label);

        std::map<int, ellipsoid*> GetAllEllipsoidsMap();

        bool AddPointCloudList(const string& name, PointCloud* pCloud, int type = 0);   // type 0: replace when exist,  type 1: add when exist
        bool DeletePointCloudList(const string& name, int type = 0);    // type 0: complete matching, 1: partial matching
        bool ClearPointCloudLists();

        std::map<string, PointCloud*> GetPointCloudList();

    protected:
        std::vector<ellipsoid*> mspEllipsoids;
        std::set<plane*> mspPlanes;

        std::mutex mMutexMap;

        g2o::SE3Quat* mCameraState;   // Twc
        std::vector<g2o::SE3Quat*> mvCameraStates;      // Twc  camera in world

        std::set<PointXYZRGB*> mspPoints;  
        std::map<string, PointCloud*> mmPointCloudLists; // name-> pClouds

    public:
        // those visual ellipsoids are for visualization only and DO NOT join the optimization
        void addEllipsoidVisual(ellipsoid* pObj);
        std::vector<ellipsoid*> GetAllEllipsoidsVisual();
        void ClearEllipsoidsVisual();

    protected:
        std::vector<ellipsoid*> mspEllipsoidsVisual;

    };
}

#endif //ELLIPSOIDSLAM_MAP_H
