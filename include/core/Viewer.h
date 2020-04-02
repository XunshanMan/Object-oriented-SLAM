#ifndef ELLIPSOIDSLAM_VIEWER_H
#define ELLIPSOIDSLAM_VIEWER_H

#include <opencv2/opencv.hpp>
#include <string>

#include "MapDrawer.h"
#include "System.h"
#include "FrameDrawer.h"
#include <mutex>


using namespace std;

struct MenuStruct
{
    double min;
    double max;
    double def;
    string name;
};

namespace EllipsoidSLAM
{
    class MapDrawer;
    class System;
    class FrameDrawer;

    class Viewer
    {
    public:
        Viewer(const string &strSettingPath, MapDrawer* pMapDrawer);
        Viewer(System* pSystem, const string &strSettingPath, MapDrawer* pMapDrawer);

        void SetFrameDrawer(FrameDrawer* p);

        void run();

        bool updateObject();
        bool updateCamera();

        void RequestFinish();
        bool isFinished();

        // dynamic menu.
        int addDoubleMenu(string name, double min, double max, double def);
        bool getValueDoubleMenu(int id, double &value);

    private:
        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        MapDrawer* mpMapDrawer;
        System* mpSystem;
        FrameDrawer* mpFrameDrawer;

        // menu lists
        vector<pangolin::Var<double>*> mvDoubleMenus;
        vector<MenuStruct> mvMenuStruct;

        // Manual control of the point cloud visualization
        map<string,pangolin::Var<bool>*> mmPointCloudOptionMenus;

        std::map<std::string,bool> mmPointCloudOptionMap;
        void RefreshPointCloudOptions();
        
        void RefreshMenu();

        int miRows;
        int miCols;


    };


}

#endif //ELLIPSOIDSLAM_VIEWER_H
