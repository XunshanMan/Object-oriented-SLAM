#include "include/core/System.h"
#include "include/utils/dataprocess_utils.h"

#include "src/config/Config.h"

namespace EllipsoidSLAM
{

    System::System(const string &strSettingsFile, const bool bUseViewer) {

        cout << endl <<
        "EllipsoidSLAM Project 2019, Beihang University." << endl;
        cout << " Input Sensor: RGB-D " << endl;

        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }
        
        // Initialize global settings.
        Config::Init();
        Config::SetParameterFile(strSettingsFile);

        //Create the Map
        mpMap = new Map();

        mpFrameDrawer = new FrameDrawer(mpMap);

        //Create Drawers. These are used by the Viewer
        mpMapDrawer = new MapDrawer(strSettingsFile, mpMap);

        mpTracker = new Tracking(this, mpFrameDrawer, mpMapDrawer, mpMap, strSettingsFile);
        mpFrameDrawer->setTracker(mpTracker);
        
        //Initialize the Viewer thread and launch
        if(bUseViewer)
        {
            mpViewer = new Viewer(this, strSettingsFile, mpMapDrawer);
            mptViewer = new thread(&Viewer::run, mpViewer);
            mpViewer->SetFrameDrawer(mpFrameDrawer);
        }

        OpenDepthEllipsoid();   // Open Single-Frame Ellipsoid Extraction
        mpTracker->OpenGroundPlaneEstimation();     // Open Groundplane Estimation.
    }

    bool System::TrackWithObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd & bboxMat, const cv::Mat &imDepth, const cv::Mat &imRGB,
                    bool withAssociation)
    {
        return mpTracker->GrabPoseAndObjects(timestamp, pose, bboxMat, imDepth, imRGB, withAssociation);
    }

    Map* System::getMap() {
        return mpMap;
    }

    MapDrawer* System::getMapDrawer() {
        return mpMapDrawer;
    }

    FrameDrawer* System::getFrameDrawer() {
        return mpFrameDrawer;
    }

    Viewer* System::getViewer() {
        return mpViewer;
    }

    Tracking* System::getTracker() {
        return mpTracker;
    }

    void System::SaveObjectsToFile(string &path){
        auto ellipsoids = mpMap->GetAllEllipsoids();
        
        MatrixXd objMat;objMat.resize(ellipsoids.size(), 11);
        int i=0;
        for(auto e : ellipsoids)
        {
            Vector10d vec = e->toVector();
            Eigen::Matrix<double, 11, 1> vec_instance;
            vec_instance << e->miInstanceID, vec;
            objMat.row(i++) = vec_instance;
        }

        saveMatToFile(objMat, path.c_str());

        std::cout << "Save " << ellipsoids.size() << " objects to " << path << std::endl;
    }

    void System::OpenDepthEllipsoid()
    {
        mpTracker->OpenDepthEllipsoid();
    }

    void System::OpenOptimization()
    {
        mpTracker->OpenOptimization();
    }

    void System::CloseOptimization()
    {
        mpTracker->CloseOptimization();
    }
}



