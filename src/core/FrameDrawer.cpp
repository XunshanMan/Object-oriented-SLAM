#include "include/core/FrameDrawer.h"
#include "src/config/Config.h"
#include "utils/dataprocess_utils.h"

using namespace cv;

namespace EllipsoidSLAM
{
    FrameDrawer::FrameDrawer(EllipsoidSLAM::Map *pMap) {

        mpMap = pMap;
        mmRGB = cv::Mat();
        mmDepth = cv::Mat();
    }

    void FrameDrawer::setTracker(EllipsoidSLAM::Tracking *pTracker) {
        mpTracking = pTracker;
    }

    cv::Mat FrameDrawer::drawFrame() {
        Frame * frame = mpTracking->mCurrFrame;
        if( frame == NULL ) return cv::Mat();
        cv::Mat im;
        if(!frame->rgb_img.empty()) // use rgb image if it exists, or use depth image instead.
            im = frame->rgb_img;
        else 
            im = frame->frame_img;

        cv::Mat out = drawFrameOnImage(im);

        mmRGB = out.clone();

        return mmRGB;
    }

    cv::Mat FrameDrawer::drawDepthFrame() {
        Frame * frame = mpTracking->mCurrFrame;
        cv::Mat I = frame->frame_img;   // U16C1 , ushort
        cv::Mat im,R,G,B;

        // Vector3d color1(51,25,0);
        // Vector3d color2(255,229,204);
        // r = color1 + value/255*(color2-color1)
        // (255-51)/255   51
        // 220-25

        I.convertTo(R, CV_8UC1, 0.028, 51);
        I.convertTo(G, CV_8UC1, 0.028, 25);
        I.convertTo(B, CV_8UC1, 0.028, 0);
        std::vector<cv::Mat> array_to_merge;
        array_to_merge.push_back(B);
        array_to_merge.push_back(G);
        array_to_merge.push_back(R);
        cv::merge(array_to_merge, im);

        cv::Mat out = drawObservationOnImage(im);

        mmDepth = out.clone();

        return mmDepth;
    }

    cv::Mat FrameDrawer::drawProjectionOnImage(cv::Mat &im) {
        std::map<int, ellipsoid*> pEllipsoidsMapWithLabel = mpMap->GetAllEllipsoidsMap();

        // draw projected bounding boxes of the ellipsoids in the map
        cv::Mat imageProj = im.clone();
        for(auto iter=pEllipsoidsMapWithLabel.begin(); iter!=pEllipsoidsMapWithLabel.end();iter++)
        {
            ellipsoid* e = iter->second;

            // check whether it could be seen
            if( e->CheckObservability(mpTracking->mCurrFrame->cam_pose_Tcw) )
            {
                Vector4d rect = e->getBoundingBoxFromProjection(mpTracking->mCurrFrame->cam_pose_Tcw, mpTracking->mCalib); // center, width, height
                cv::rectangle(imageProj, cv::Rect(cv::Point(rect[0],rect[1]),cv::Point(rect[2],rect[3])), cv::Scalar(0,0,255), 4);
            }
        }

        return imageProj.clone();
    }

    cv::Mat FrameDrawer::drawFrameOnImage(cv::Mat &in) {
        cv::Mat out = in.clone();
        out = drawObservationOnImage(out);
        out = drawProjectionOnImage(out);
        return out.clone();
    }

    cv::Mat FrameDrawer::drawObservationOnImage(cv::Mat &in) {
        cv::Mat im = in.clone();

        Frame * frame = mpTracking->mCurrFrame;
        // draw observation
        Eigen::MatrixXd mat_det = frame->mmObservations;
        int obs = mat_det.rows();
        for(int r=0;r<obs;r++){
            VectorXd vDet = mat_det.row(r);

            Vector4d measure; measure << vDet(1), vDet(2), vDet(3), vDet(4);
            bool is_border = calibrateMeasurement(measure, im.rows, im.cols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));

            int labelId = int(vDet(5));

            Rect rec(Point(vDet(1), vDet(2)), Point(vDet(3), vDet(4)));

            if( !is_border )
            {
                rectangle(im, rec, Scalar(255,0,0), 3);
                putText(im, to_string(labelId), Point(vDet(1), vDet(2)), cv::FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0), 2);
            }
        }

        return im.clone();
    }

    cv::Mat FrameDrawer::getCurrentFrameImage(){
        return mmRGB;
    }

    cv::Mat FrameDrawer::getCurrentDepthFrameImage(){
        return mmDepth;
    }
}