#include <include/core/Frame.h>

namespace EllipsoidSLAM
{
    int Frame::total_frame=0;

    Frame::Frame(double timestamp_, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB)
    {
        timestamp = timestamp_;
        rgb_img = imRGB.clone();
        frame_img = imDepth.clone();

        cam_pose_Twc.fromVector(pose.tail(7));
        cam_pose_Tcw = cam_pose_Twc.inverse();

        frame_seq_id = total_frame++;

        mvbOutlier.resize(bboxMap.rows());
        fill(mvbOutlier.begin(), mvbOutlier.end(), false);  

        mmObservations = bboxMap;

        std::cout << "--------> New Frame : " << frame_seq_id << " Timestamp: " << std::to_string(timestamp) << std::endl;
        std::cout << "[Frame.cpp] bboxMap : " << std::endl << bboxMap << std::endl << std::endl;

        mbHaveLocalObject = false;
    }


}
