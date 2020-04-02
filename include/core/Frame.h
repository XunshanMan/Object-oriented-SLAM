#ifndef ELLIPSOIDSLAM_FRAME_H
#define ELLIPSOIDSLAM_FRAME_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Ellipsoid.h"

namespace EllipsoidSLAM{
    
class SymmetryOutputData;

class Frame{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &im);
    Frame(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB);
    Frame(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB);

    int static total_frame;

    int frame_seq_id;    // image topic sequence id, fixed
    cv::Mat frame_img;      // depth img for processing
    cv::Mat rgb_img;        // rgb img for visualization.
    cv::Mat ellipsoids_2d_img;

    double timestamp;

    g2o::VertexSE3Expmap* pose_vertex;

    g2o::SE3Quat cam_pose_Tcw;	     // optimized pose  world to cam
    g2o::SE3Quat cam_pose_Twc;	     // optimized pose  cam to world

    Eigen::MatrixXd mmObservations;     // id x1 y1 x2 y2 label rate instanceID
    std::vector<bool> mvbOutlier;

    // For depth ellipsoid extraction.
    bool mbHaveLocalObject;
    std::vector<g2o::ellipsoid*> mpLocalObjects; // local 3d ellipsoid
};

}
#endif //ELLIPSOIDSLAM_FRAME_H
