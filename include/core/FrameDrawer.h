#ifndef ELLIPSOIDSLAM_FRAMEDRAWER_H
#define ELLIPSOIDSLAM_FRAMEDRAWER_H

#include "Map.h"
#include "Tracking.h"

namespace EllipsoidSLAM{

class Map;
class Tracking;

class FrameDrawer {
public:
    FrameDrawer(Map* pMap);

    void setTracker(Tracking* pTracker);

    cv::Mat drawFrame();
    cv::Mat drawFrameOnImage(cv::Mat &im);

    cv::Mat drawDepthFrame();

    cv::Mat getCurrentFrameImage();
    cv::Mat getCurrentDepthFrameImage();

private:

    Map* mpMap;
    Tracking* mpTracking;

    cv::Mat drawProjectionOnImage(cv::Mat &im);
    cv::Mat drawObservationOnImage(cv::Mat &im);

    cv::Mat mmRGB;
    cv::Mat mmDepth;
};

}
#endif //ELLIPSOIDSLAM_FRAMEDRAWER_H
