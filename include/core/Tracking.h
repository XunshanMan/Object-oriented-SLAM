#ifndef ELLIPSOIDSLAM_TRACKING_H
#define ELLIPSOIDSLAM_TRACKING_H

#include "System.h"
#include "FrameDrawer.h"
#include "Viewer.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Initializer.h"
#include "Optimizer.h"

#include <src/symmetry/Symmetry.h>
#include <src/pca/EllipsoidExtractor.h>
#include <src/plane/PlaneExtractor.h>
#include "DataAssociation.h"
#include <src/dense_builder/builder.h>

namespace EllipsoidSLAM{

class System;
class FrameDrawer;
class MapDrawer;
class Map;
class Viewer;
class Frame;
class Initializer;
class Optimizer;
class Symmetry;

class Tracking {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Tracking(System* pSys, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             const string &strSettingPath);

    bool GrabPoseAndSingleObjectAnnotation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection);
    bool GrabPoseAndObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB = cv::Mat(), bool withAssociation = false);
    bool GrabPoseAndObjects(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB = cv::Mat(), bool withAssociation = false);

    Frame* mCurrFrame;

    Eigen::Matrix3d mCalib;

    void outputBboxMatWithAssociation();

    void SaveObjectHistory(const string& path);

    void OpenOptimization();
    void CloseOptimization();

    bool SavePointCloudMap(const string& path);

    std::vector<bool> checkKeyFrameForInstances(std::vector<int>& associations);

    // Single-frame Ellipsoid Extraction
    void OpenDepthEllipsoid();

    // Groundplane Estimation
    void OpenGroundPlaneEstimation();
    void CloseGroundPlaneEstimation();
    int GetGroundPlaneEstimationState();

private:

    void ProcessCurrentFrame(bool withAssociation);

    void UpdateObjectObservation(Frame* pFrame, bool withAssociation = false);

    void JudgeInitialization();

    bool isKeyFrameForVisualization(); 

    void ProcessVisualization();

    void RefreshObjectHistory();
    void ProcessGroundPlaneEstimation();
    void Update3DObservationDataAssociation(EllipsoidSLAM::Frame* pFrame, std::vector<int>& associations, std::vector<bool>& KeyFrameChecks);
    void UpdateDepthEllipsoidEstimation(EllipsoidSLAM::Frame* pFrame, bool withAssociation);

    std::vector<int> GetMannualAssociation(Eigen::MatrixXd &obsMat);

protected:

    g2o::ellipsoid* getObjectDataAssociation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection);

    // System
    System* mpSystem;

    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    Initializer* mpInitializer;

    //Map
    Map* mpMap;

    // Optimizer
    Optimizer* mpOptimizer;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    camera_intrinsic mCamera;

    int mRows, mCols;

    std::vector<Frame*> mvpFrames;

    // Store observations in a map with instance id.
    // In the future, storing observations under Ellipsoid class separately would make it clearer.
    std::map<int, Observations> mmObjectObservations;

    Builder* mpBuilder;     // a dense pointcloud builder from visualization

    std::map<int, MatrixXd> mmObjectHistory;

    bool mbOpenOptimization;

    bool mbDepthEllipsoidOpened;
    EllipsoidExtractor* mpEllipsoidExtractor;
    std::map<int, Observation3Ds> mmObjectObservations3D;  // 3d observations indexed by instance ID

    DataAssociationSolver* pDASolver;

    int miGroundPlaneState; // 0: Closed  1: estimating 2: estimated
    g2o::plane mGroundPlane;
    PlaneExtractor* pPlaneExtractor;
};

}

#endif //ELLIPSOIDSLAM_TRACKING_H
