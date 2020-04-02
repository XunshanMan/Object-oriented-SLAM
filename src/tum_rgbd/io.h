// Basic input and output of the TUM-RGBD dataset.

#include <string>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <map>

#include <core/Ellipsoid.h>  

using namespace std;
using namespace cv;
using namespace Eigen;

namespace TUMRGBD
{

/*
*   All the timestamp is based on RGB images.
*/
class Dataset
{
    public:
        // Load the next RGB-D frame containing a RGB image, a depth image and a camera pose.
        // pose: x y z qx qy qz qw
        bool readFrame(cv::Mat &rgb, cv::Mat &depth, Eigen::VectorXd &pose);
        void loadDataset(string &path);
        std::vector<int> generateValidVector();

        // load object detections
        bool loadDetectionDir(string &path);
        Eigen::MatrixXd getDetectionMat();

        bool empty();

        int getCurrentID();
        int getTotalNum();

        // load a specified frame using its ID
        bool findFrameUsingID(int id, cv::Mat &rgb, cv::Mat &depth, Eigen::VectorXd &pose);
        
        // load an odometry data generated from wheels or visual odometry algorithm like ORB-SLAM.
        // the system will automatically calibrate its coordinate by aligning the pose of the first frame to the corresponding ground truth.
        bool SetOdometry(const string& dir_odom, bool calibrate = true);

        // jump to a specified frame using its ID
        void SetCurrentID(int id);

        // Get the timestamp of the current frame
        double GetCurrentTimestamp();
        double GetTimestamp(int id);

        bool getPoseFromTimeStamp(string &timestamp, VectorXd &pose);

    private:
        void associateRGBWithGroundtruth();

        void loadGroundTruthToMap(string &path);

        // load the associations from the timestamps of rgb images to the timestamps of depth images
        void LoadAssociationRGBToDepth(string &path);
        void LoadAssociationRGBToGroundtruth(string &path);

        void generateIndexIdToRGBTimeStamp();

        bool judgeValid(int id);        // the frame with the id is valid when it contains valid depth and rgb images ...

        bool getPoseFromRGBTimeStamp(string &timestamp, VectorXd &pose);
        VectorXd calibratePose(VectorXd& pose);   // calibrate the odom data by aligning to the first frame of the groundtruth 
    private:
        string msDatasetDir;   // the root directory of the dataset

        string msRGBDir; 
        string msDepthDir;
        string msGroundtruthPath;

        string msAssociatePath;
        string msAssociateGroundtruthPath;

        string msDetectionDir;

        vector<string> mvRGBFileNames;   // store the full paths of all the rgb images 

        vector<VectorXd> mvPoses;

        int miCurrentID;
        int miTotalNum;

        vector<string> mvIdToGroundtruthTimeStamp;     
        vector<string> mvIdToDepthTimeStamp;  
        vector<string> mvIdToDepthImagePath;  
        vector<string> mvIdToRGBTimeStamp;

        map<string, VectorXd> mmTimeStampToPose;
        map<string, VectorXd> mmOdomRGBStampToPose;

        bool mbDetectionLoaded;

        bool mbOdomSet;
        string msOdomDir;
        g2o::SE3Quat* mTransGtCalibrate;
};

}