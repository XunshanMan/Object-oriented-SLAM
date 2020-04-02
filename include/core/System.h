#ifndef ELLIPSOIDSLAM_SYSTEM_H
#define ELLIPSOIDSLAM_SYSTEM_H

#include "Frame.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Viewer.h"
#include "Tracking.h"
#include "FrameDrawer.h"

#include <Eigen/Core>
#include <string>
#include <thread>

namespace EllipsoidSLAM{

class Viewer;
class Frame;
class Map;
class Tracking;
class FrameDrawer;

class System {


public:
    System(const string &strSettingsFile, const bool bUseViewer = true);

    // Interface.
    // timestamp: sec
    // pose: camera pose in the world coordinate ; x y z qx qy qz qw
    // imDepth: depth image; CV16UC1
    // imRGB: rgb image for visualization; CV8UC3, BGR
    // bboxMat: id x1 y1 x2 y2 label prob [Instance]
    // withAssociation: 
    //      true, use [instance] as association result; false, use single-frame ellipsoid estimation to solve data association
    bool TrackWithObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd & bboxMat, const cv::Mat &imDepth, const cv::Mat &imRGB = cv::Mat(),
                        bool withAssociation = false);

    Map* getMap();
    MapDrawer* getMapDrawer();
    FrameDrawer* getFrameDrawer();
    Viewer* getViewer();
    Tracking* getTracker();

    // save objects to file
    void SaveObjectsToFile(string &path);

    void OpenDepthEllipsoid();

    void OpenOptimization();
    void CloseOptimization();

private:

    Map* mpMap;

    Viewer* mpViewer;

    MapDrawer* mpMapDrawer;

    Tracking* mpTracker;
    FrameDrawer* mpFrameDrawer;

    std::thread* mptViewer;

};

}   // namespace EllipsoidSLAM

#endif //ELLIPSOIDSLAM_SYSTEM_H
