#include "core/Initializer.h"
#include "core/Geometry.h"
#include "utils/dataprocess_utils.h"
#include "utils/matrix_utils.h"

#include <Eigen/Core>

#include "include/core/Viewer.h"
#include "include/core/MapDrawer.h"
#include "include/core/Map.h"

#include <thread>
#include <string>

#include "include/core/Ellipsoid.h"
#include "src/tum_rgbd/io.h"

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudPCL;

#include "src/config/Config.h"

using namespace std;
using namespace Eigen;

int main(int argc,char* argv[]) {

    if( argc != 3)
    {
        std::cout << "usage: " << argv[0] << " path_to_settings path_to_dataset" << std::endl;
        return 1;
    }
    string strSettingPath = string(argv[1]);
    string dataset_path(argv[2]);
    string strDetectionDir = dataset_path + "bbox/";

    std::cout << "- settings file: " << strSettingPath << std::endl;
    std::cout << "- dataset_path: " << dataset_path << std::endl;
    std::cout << "- strDetectionDir: " << strDetectionDir << std::endl;

    TUMRGBD::Dataset loader;
    loader.loadDataset(dataset_path);
    loader.loadDetectionDir(strDetectionDir);

    EllipsoidSLAM::System SLAM(strSettingPath, true);

    cv::Mat rgb,depth;
    VectorXd pose;
    while(!loader.empty())
    {
        bool valid = loader.readFrame(rgb,depth,pose);
        int current_id = loader.getCurrentID();

        Eigen::MatrixXd detMat = loader.getDetectionMat();
        double timestamp = loader.GetCurrentTimestamp();

        if(valid)
        {
            std::cout << "*****************************" << std::endl;
            std::cout << "Press [ENTER] to continue ... " << std::endl;
            std::cout << "*****************************" << std::endl;
            getchar();

            SLAM.TrackWithObjects(timestamp, pose, detMat, depth, rgb, true);        // Process frame.    
            std::cout << std::endl;
        }

        std::cout << " -> " << loader.getCurrentID() << "/" << loader.getTotalNum() << std::endl;
    }

    std::cout << "Finished all data." << std::endl;

    // save objects 
    string output_path("./objects.txt");
    SLAM.SaveObjectsToFile(output_path);
    SLAM.getTracker()->SaveObjectHistory("./object_history.txt");

    cout << "Use Ctrl+C to quit." << endl;
    while(1);

    cout << "End." << endl;
    return 0;
}