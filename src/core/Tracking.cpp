#include "include/core/Tracking.h"
#include "src/config/Config.h"
#include "utils/dataprocess_utils.h"

Eigen::MatrixXd matSymPlanes;

namespace EllipsoidSLAM
{
    void outputObjectObservations(std::map<int, Observations> &mmObjectObservations)
    {
        ofstream out("./log_mmObjectObservations.txt");
        
        out << " --------- ObjectObservations : " << std::endl;
        for ( auto obPair: mmObjectObservations)
        {
            out << " ---- Instance " << obPair.first << " (" << obPair.second.size() << ") :" << std::endl;

            for( auto ob : obPair.second )
            {
                out << " -- ob : " << ob->pFrame->frame_seq_id << " | " << ob->bbox.transpose() << " | " << ob->label << " | " << ob->rate << std::endl;
            }

            out << std::endl;
        }

        out.close();
        std::cout << "Save to log_mmObjectObservations.txt..." << std::endl;
    }

    void Tracking::outputBboxMatWithAssociation()
    {
        std::map<double, Observations> mapTimestampToObservations;

        for ( auto obPair: mmObjectObservations)
        {
            int instance = obPair.first;
            for( auto ob : obPair.second )
            {
                ob->instance = instance;

                // save with timestamp
                if(mapTimestampToObservations.find(ob->pFrame->timestamp)!=mapTimestampToObservations.end())
                    mapTimestampToObservations[ob->pFrame->timestamp].push_back(ob);
                else {
                    mapTimestampToObservations.insert(make_pair(ob->pFrame->timestamp, Observations()));
                    mapTimestampToObservations[ob->pFrame->timestamp].push_back(ob);
                }
            }
        }

        for( auto frameObsPair : mapTimestampToObservations ){
            string str_timestamp = to_string(frameObsPair.first);

            string filename = string("./bbox/") + str_timestamp + ".txt";
            ofstream out(filename.c_str());
            
            int num = 0;
            for ( auto ob: frameObsPair.second)
            {
                
                out << num++ << " " << ob->bbox.transpose() << " " << ob->label << " " << ob->rate << " " << ob->instance << std::endl;
            }

            out.close();
            std::cout << "Save to " << filename << std::endl;
        }

        std::cout << "Finish... " << std::endl;
                
    }

    Tracking::Tracking(EllipsoidSLAM::System *pSys, EllipsoidSLAM::FrameDrawer *pFrameDrawer,
                       EllipsoidSLAM::MapDrawer *pMapDrawer, EllipsoidSLAM::Map *pMap, const string &strSettingPath)
                       :mpMap(pMap), mpSystem(pSys), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer)
   {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=30;

        int rows = fSettings["Camera.height"];
        int cols = fSettings["Camera.width"];

        mCalib << fx,  0,  cx,
                0,  fy, cy,
                0,      0,     1;

        mCamera.cx = cx;
        mCamera.cy = cy;
        mCamera.fx = fx;
        mCamera.fy = fy;
        mCamera.scale = fSettings["Camera.scale"];

        mpInitializer =  new Initializer(rows, cols);
        mpOptimizer = new Optimizer;
        mRows = rows;
        mCols = cols;

        mbDepthEllipsoidOpened = false;

        mbOpenOptimization = true;

        pDASolver = new DataAssociationSolver(mpMap);
        mpBuilder = new Builder();
        mpBuilder->setCameraIntrinsic(mCalib, mCamera.scale);

        mCurrFrame = NULL;

        // output
        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if(DistCoef.rows==5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;
        cout << "- rows: " << rows << endl;
        cout << "- cols: " << cols << endl;
        cout << "- Scale: " << mCamera.scale << endl;

        // ********** DEBUG ***********
        matSymPlanes.resize(0, 5);
    }

    g2o::ellipsoid* Tracking::getObjectDataAssociation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection) {
        auto objects = mpMap->GetAllEllipsoids();
        if( objects.size() > 0 )
            return objects[0];
        else
            return NULL;
    }

    bool Tracking::GrabPoseAndObjects(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
    const cv::Mat &imDepth, const cv::Mat &imRGB, bool withAssociation) {
        return GrabPoseAndObjects(0, pose, bboxMap, imDepth, imRGB, withAssociation);
    }

    bool Tracking::GrabPoseAndObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
        const cv::Mat &imDepth, const cv::Mat &imRGB, bool withAssociation) {

        clock_t time_start = clock();        

        Frame *pF = new Frame(timestamp, pose, bboxMap, imDepth, imRGB);
        mvpFrames.push_back(pF);
        mCurrFrame = pF;

        clock_t time_init_frame_process = clock();

        UpdateObjectObservation(mCurrFrame, withAssociation);   // Store object observation in a specific data structure.
        clock_t time_UpdateObjectObservation = clock();

        JudgeInitialization();

        if(mbOpenOptimization){
            ProcessCurrentFrame(withAssociation);
        }
        clock_t time_ProcessCurrentFrame = clock();

        // Visualization
        ProcessVisualization();
        clock_t time_Visualization = clock();

        // // Output running time
        // cout << " - System Time: " << endl;
        // cout << " -- time_init_frame_process: " <<(double)(time_init_frame_process - time_start) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " -- time_UpdateObjectObservation: " <<(double)(time_UpdateObjectObservation - time_init_frame_process) / CLOCKS_PER_SEC << "s" << endl;
        // cout << " -- time_ProcessCurrentFrame: " <<(double)(time_ProcessCurrentFrame - time_UpdateObjectObservation) / CLOCKS_PER_SEC << "s" << endl;
        // cout << " -- time_Visualization: " <<(double)(time_Visualization - time_ProcessCurrentFrame) / CLOCKS_PER_SEC << "s" << endl;
        // cout << " - [ total_frame: " <<(double)(time_Visualization - time_start) / CLOCKS_PER_SEC << "s ]" << endl;

        return true;
    }

    void Tracking::ProcessVisualization()
    {
        // Visualize frames with intervals
        if( isKeyFrameForVisualization() )
            mpMap->addCameraStateToTrajectory(&mCurrFrame->cam_pose_Twc);
        mpMap->setCameraState(&mCurrFrame->cam_pose_Twc);

        // Render rgb images and depth images for visualization.
        cv::Mat imForShow = mpFrameDrawer->drawFrame();
        cv::Mat imForShowDepth = mpFrameDrawer->drawDepthFrame();
        
    }

    void Tracking::ProcessCurrentFrame(bool withAssociation){
        clock_t time_start = clock();        

        double depth_range = Config::ReadValue<double>("EllipsoidExtractor_DEPTH_RANGE");   // Only consider pointcloud within depth_range

        // Begin Global Optimization when there are objects in map.
        if(mpMap->GetAllEllipsoids().size()>0){
            mpOptimizer->GlobalObjectGraphOptimization(mvpFrames, mpMap, mRows, mCols, mCalib, mmObjectObservations, true, withAssociation);            

            RefreshObjectHistory();
        }
        clock_t time_optimization = clock();        

        // [A visualization tool] When Builder is opened, it generates local pointcloud from depth and rgb images of current frame,
        // and global pointcloud by simply adding local pointcloud in world coordinate and then downsampling them for visualization. 
        bool mbOpenBuilder = Config::Get<int>("Visualization.Builder.Open") == 1;
        if(mbOpenBuilder)
        {
            if(!mCurrFrame->rgb_img.empty()){    // RGB images are needed.
                Eigen::VectorXd pose = mCurrFrame->cam_pose_Twc.toVector();
                mpBuilder->processFrame(mCurrFrame->rgb_img, mCurrFrame->frame_img, pose, depth_range);

                mpBuilder->voxelFilter(0.01);   // Down sample threshold; smaller the finer; depend on the hardware.
                PointCloudPCL::Ptr pCloudPCL = mpBuilder->getMap();
                PointCloudPCL::Ptr pCurrentCloudPCL = mpBuilder->getCurrentMap();

                auto pCloud = pclToQuadricPointCloudPtr(pCloudPCL);
                auto pCloudLocal = pclToQuadricPointCloudPtr(pCurrentCloudPCL);
                mpMap->AddPointCloudList("Builder.Global Points", pCloud);
                mpMap->AddPointCloudList("Builder.Local Points", pCloudLocal);
            }
        }
        clock_t time_builder = clock();        

        // // Output running time
        // cout << " -- ProcessCurrentFrame Time: " << endl;
        // cout << " --- time_optimization: " <<(double)(time_optimization - time_start) / CLOCKS_PER_SEC << "s" << endl;
        // cout << " --- time_builder: " <<(double)(time_builder - time_optimization) / CLOCKS_PER_SEC << "s" << endl;
    }

    void AddSegCloudsToQuadricStorage(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& segClouds, EllipsoidSLAM::PointCloud* pSegCloud){
        int cloud_num = segClouds.size();
        srand(time(0));
        for(int i=0;i<cloud_num;i++)
        {
            int point_num = segClouds[i]->points.size();
            int r = rand()%155;
            int g = rand()%155;
            int b = rand()%155;
            for( int n=0;n<point_num;n++)
            {
                PointXYZRGB point;
                point.x =  segClouds[i]->points[n].x;
                point.y =  segClouds[i]->points[n].y;
                point.z =  segClouds[i]->points[n].z;
                point.r = r;
                point.g = g;
                point.b = b;

                point.size = 2;
                pSegCloud->push_back(point);
            }
            
        }

        return;

    }

    // Process Ellipsoid Estimation for every boundingboxes in current frame.
    // Finally, store 3d Ellipsoids into the membter variable mpLocalObjects of pFrame.
    void Tracking::UpdateDepthEllipsoidEstimation(EllipsoidSLAM::Frame* pFrame, bool withAssociation)
    {
        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int rows = obs_mat.rows();

        Eigen::VectorXd pose = pFrame->cam_pose_Twc.toVector();
        EllipsoidSLAM::PointCloud* pCenterCloud = new EllipsoidSLAM::PointCloud;
        EllipsoidSLAM::PointCloud* pSegCloud = new EllipsoidSLAM::PointCloud;

        mpEllipsoidExtractor->ClearPointCloudList();    // clear point cloud visualization

        bool bPlaneNotClear = true;
        for(int i=0;i<rows;i++){
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate instanceID
            int label = round(det_vec(5));

            Eigen::Vector4d measurement = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            // Filter those detections lying on the border.
            bool is_border = calibrateMeasurement(measurement, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));

            g2o::ellipsoid* pE_extracted = NULL;
            // 2 conditions must meet to start ellipsoid extraction:
            // C1 : the bounding box is not on border
            bool c1 = !is_border;

            // C2 : the groundplane has been estimated successfully
            bool c2 = miGroundPlaneState == 2;
            
            // in condition 3, it will not start
            // C3 : under with association mode, and the association is invalid, no need to extract ellipsoids again.
            bool c3 = false;
            if( withAssociation )
            {
                int instance = round(det_vec(7));
                if ( instance < 0 ) c3 = true;  // invalid instance
            }
            
            if( c1 && c2 && !c3 ){   
                g2o::ellipsoid e_extractByFitting_newSym = mpEllipsoidExtractor->EstimateLocalEllipsoid(pFrame->frame_img, measurement, label, pose, mCamera);

                if(bPlaneNotClear){
                    mpMap->clearPlanes();
                    if(miGroundPlaneState == 2) // if the groundplane has been estimated
                        mpMap->addPlane(&mGroundPlane);
                    bPlaneNotClear = false;
                }

                if( mpEllipsoidExtractor->GetResult() )
                {
                    g2o::ellipsoid *pE_extractByFitting = new g2o::ellipsoid(e_extractByFitting_newSym);
                    // std::cout << "[Tracking.cpp] Get an Ellipsoid [fixSym]: " << e_extractByFitting_newSym.toMinimalVector().transpose() << std::endl;

                    pE_extracted = pE_extractByFitting;   // Store result to pE_extracted.

                    // Visualize estimated ellipsoid
                    g2o::ellipsoid* pObjByFitting = new g2o::ellipsoid(e_extractByFitting_newSym.transform_from(pFrame->cam_pose_Twc));
                    pObjByFitting->setColor(Vector3d(0.8,0.0,0.0), 1); // Set green color
                    mpMap->addEllipsoidVisual(pObjByFitting);
                    
                    // Visualize symmetry plane
                    SymmetryOutputData symOutputData = mpEllipsoidExtractor->GetSymmetryOutputData();
                    if(symOutputData.result){
                        Vector3d planeCenter = pObjByFitting->pose.translation();
                        if(symOutputData.symmetryType == 1){
                            g2o::plane* pSymPlane = new g2o::plane(symOutputData.planeVec, Vector3d(1.0,0,0.0));
                            pSymPlane->InitFinitePlane(planeCenter, 1); 
                            mpMap->addPlane(pSymPlane);
                        }
                        else if(symOutputData.symmetryType == 2) // dual reflection 
                        {
                            g2o::plane* pSymPlane = new g2o::plane(symOutputData.planeVec, Vector3d(0.0,0.8,0.0));
                            pSymPlane->InitFinitePlane(planeCenter, 1); 
                            mpMap->addPlane(pSymPlane);

                            g2o::plane* pSymPlane2 = new g2o::plane(symOutputData.planeVec2, Vector3d(0.0,0.8,0));
                            pSymPlane2->InitFinitePlane(planeCenter, 1); 
                            mpMap->addPlane(pSymPlane2);
                        }

                        // output ;  Save every estimated symmetry parameter for every frame.
                        VectorXd symPlaneVec; symPlaneVec.resize(5);   // Param(4) ; error
                        symPlaneVec << symOutputData.planeVec, symOutputData.prob;
                        addVecToMatirx(matSymPlanes, symPlaneVec);
                    }

                }   // successful estimation.

            }
            pFrame->mpLocalObjects.push_back(pE_extracted);
        }
    }

    void Tracking::Update3DObservationDataAssociation(EllipsoidSLAM::Frame* pFrame, std::vector<int>& associations, std::vector<bool>& KeyFrameChecks)
    {
        int num = associations.size();

        if( mbDepthEllipsoidOpened )
        {
            std::vector<g2o::ellipsoid*> pLocalObjects = pFrame->mpLocalObjects;

            for( int i=0; i<num; i++)
            {
                if(pLocalObjects[i] == NULL )   // if the single-frame ellipsoid estimation fails
                    continue;

                int instance = associations[i];
                if(instance < 0 ) continue; // if the data association is invalid

                if( !KeyFrameChecks[i] )  // if the observation for the object is not key observation (without enough intervals to the last observation).
                {
                    pFrame->mpLocalObjects[i] = NULL;
                    continue;
                }

                // Save 3D observations
                Observation3D* pOb3d = new Observation3D;
                pOb3d->pFrame = pFrame;
                pOb3d->pObj = pLocalObjects[i];
                mmObjectObservations3D[instance].push_back(pOb3d);

                // Set instance to the ellipsoid according to the associations
                pLocalObjects[i]->miInstanceID = instance;
            }
        }

        return;
    }

    // Consider key observations for every object instances.
    // key observations: two valid observations for the same instance should have enough intervals( distance or angles between the two poses ).
    std::vector<bool> Tracking::checkKeyFrameForInstances(std::vector<int>& associations)
    {
        double CONFIG_KEYFRAME_DIS;
        double CONFIG_KEYFRAME_ANGLE;

        if( Config::Get<int>("Tracking.KeyFrameCheck.Close") == 1)
        {
            CONFIG_KEYFRAME_DIS = 0;  
            CONFIG_KEYFRAME_ANGLE = 0; 
        }
        else
        {
            CONFIG_KEYFRAME_DIS = 0.4;  
            CONFIG_KEYFRAME_ANGLE = CV_PI/180.0*15; 
        }

        int num =associations.size();
        std::vector<bool> checks; checks.resize(num);
        fill(checks.begin(), checks.end(), false);
        for( int i=0;i<num;i++)
        {
            int instance = associations[i];
            if(instance<0) 
            {
                checks[i] = false;
            }
            else
            {
                if(mmObjectObservations.find(instance) == mmObjectObservations.end())   // if the instance has not been initialized
                {
                    checks[i] = true;
                }
                else
                {
                    Observations &obs = mmObjectObservations[instance];
                    // Get last frame
                    g2o::SE3Quat &pose_last_wc = obs.back()->pFrame->cam_pose_Twc;
                    g2o::SE3Quat &pose_curr_wc = mCurrFrame->cam_pose_Twc;

                    g2o::SE3Quat pose_diff = pose_curr_wc.inverse() * pose_last_wc;
                    double dis = pose_diff.translation().norm();

                    Eigen::Quaterniond quat = pose_diff.rotation();
                    Eigen::AngleAxisd axis(quat);
                    double angle = axis.angle();

                    if( dis > CONFIG_KEYFRAME_DIS || angle > CONFIG_KEYFRAME_ANGLE)
                        checks[i] = true;
                    else
                        checks[i] = false;
                }
            }
        }
        return checks;
    }

    // for the mannual data association, 
    // this function will directly return the results of [instance] in the object detection matrix
    // PS. one row of detection matrix is : id x1 y1 x2 y2 label rate instanceID
    std::vector<int> Tracking::GetMannualAssociation(Eigen::MatrixXd &obsMat)
    {
        int num = obsMat.rows();
        std::vector<int> associations; associations.resize(num);
        for( int i=0; i<num; i++)
        {
            VectorXd vec = obsMat.row(i);
            associations[i] = round(vec[7]);
        }
        
        return associations;
    }

    void Tracking::UpdateObjectObservation(EllipsoidSLAM::Frame *pFrame, bool withAssociation) {
        mpMap->ClearEllipsoidsVisual(); // Clear the Visual Ellipsoids in the map

        // [1] Process 3d observations
        // 1.1 process groundplane estimation
        if(miGroundPlaneState == 1) // State 1: Groundplane estimation opened, and not done yet.
            ProcessGroundPlaneEstimation();

        // 1.2 process single-frame ellipsoid estimation
        if( mbDepthEllipsoidOpened )
            UpdateDepthEllipsoidEstimation(pFrame, withAssociation);

        // 1.3 process data association
        //      if data association is given, directly pass on; 
        //      if not, the DASolver will solve it automatically by comparing the estimated ellipsoid with those ellipsoids in the map. 
        //          This function is not steady and still under Test.
        std::vector<int> associations;
        if(withAssociation)
        {
            associations = GetMannualAssociation(pFrame->mmObservations);
        }
        else
        {
            assert( mbDepthEllipsoidOpened && "ATTENTION: ONLY 3D MODE IS SUPPORTED FOR AUTOMATIC DATA ASSOCIATION NOW." );
            associations = pDASolver->Solve(pFrame, mbDepthEllipsoidOpened);
        } 
        
        std::vector<bool> KeyFrameChecks = checkKeyFrameForInstances(associations);    // Check whether they are key observations

        // Save data associations to the member variable of pFrame
        Update3DObservationDataAssociation(pFrame, associations, KeyFrameChecks);
        
        // [2] Process 2D observations
        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int rows = obs_mat.rows();

        // load parameters
        int config_border_pixels = Config::Get<int>("Measurement.Border.Pixels");
        int config_lengthlimit_pixels = Config::Get<int>("Measurement.LengthLimit.Pixels");
        for(int i=0;i<rows;i++){
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate imageID
            int label = int(det_vec(5));
            int instance = associations[i];
            if(instance < 0 ) continue; // Ignore invalid associations.

            if( !KeyFrameChecks[i] )  continue; // Ignore those observations without enough intervals relative to the last observation.

            Eigen::Vector4d measurement = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));
            bool is_border = calibrateMeasurement(measurement, mRows, mCols, config_border_pixels, config_lengthlimit_pixels);

            // Ignore those measurements lying on borders.
            if( is_border ) {
                // std::cout << " [ Ignore Border detection ] " << measurement.transpose() << std::endl;
                continue;
            }

            Observation* pOb = new Observation();
            pOb->label = label;
            pOb->bbox = measurement;
            pOb->rate = det_vec(6);
            pOb->pFrame = pFrame;

            // Save 2d observations in mmObjectObservations, which will be loaded into the Optimization.
            if( mmObjectObservations.find(instance) != mmObjectObservations.end())
                mmObjectObservations[instance].push_back(pOb);
            else{
                Observations obs;
                obs.push_back(pOb);
                mmObjectObservations.insert(make_pair(instance, obs));
            }
        }
    }

    // Initilize those instances that have enough observations.
    void Tracking::JudgeInitialization() {

        // 1. Consider 2d initialization
        int CONFIG_MINIMUM_INITIALIZATION_FRAME = Config::ReadValue<int>("Tracking_MINIMUM_INITIALIZATION_FRAME");

        std::vector<ellipsoid*> objects = mpMap->GetAllEllipsoids();

        std::set<int> existInstances;   // Save the existed instances in the map
        for(auto iter=objects.begin(); iter!=objects.end(); iter++)
        {
            existInstances.insert((*iter)->miInstanceID); 
        }

        // outputObjectObservations(mmObjectObservations);
        for(auto iter=mmObjectObservations.begin(); iter!=mmObjectObservations.end(); iter++)
        {
            if( existInstances.find(iter->first) == existInstances.end() )
            {
                // if the instance has not been initialized
                Observations obs = iter->second;
                int config_frame_num = obs.size();
                if(config_frame_num < CONFIG_MINIMUM_INITIALIZATION_FRAME) continue;    // if there are enough observations

                ellipsoid e = mpInitializer->initializeQuadric(obs, mCalib);    // initialize quadrics by SVD
                e.miInstanceID = iter->first;  

                if( mpInitializer->getInitializeResult() ) {
                    ellipsoid* pBox = new ellipsoid(e);
                    mpMap->addEllipsoid(pBox);

                    // output
                    cout << std::endl;
                    cout << "-------- INITIALIZE NEW OBJECT BY SVD ---------" << endl;
                    cout << "Label id: " << pBox->miLabel << endl;
                    cout << "Instance id: " << pBox->miInstanceID << endl;
                    cout << "Initialization box: " << pBox->vec_minimal.transpose() << endl;
                    cout << std::endl;

                    continue;
                }
            }
        }

        // 2. Consider 3d initilization

        // Refresh exsited instance id.
        objects = mpMap->GetAllEllipsoids();
        existInstances.clear();
        for(auto iter=objects.begin(); iter!=objects.end(); iter++)
        {
            existInstances.insert((*iter)->miInstanceID); 
        }

        // Try initialization using single-frame ellipsoid estimation.
        if( mbDepthEllipsoidOpened ){            
            srand(time(0));
            for(auto objPair : mmObjectObservations3D)
            {
                int instance  = objPair.first;
                if( existInstances.find(instance) == existInstances.end() ) // if the instance has not been initialized yet
                {
                    Observation3D* pOb3D = objPair.second.back();
                    g2o::ellipsoid* pE = pOb3D->pObj; 
                    Frame* pFrame = pOb3D->pFrame;

                    g2o::ellipsoid* pBox = new ellipsoid(pE->transform_from(pFrame->cam_pose_Twc));
                    pBox->setColor(Vector3d(0, 0, 1));
                    mpMap->addEllipsoid(pBox);
                }
            }
        }
    }

    void Tracking::OpenDepthEllipsoid(){
        mbDepthEllipsoidOpened = true;

        mpEllipsoidExtractor = new EllipsoidExtractor;

        // Open visualization during the estimation process
        mpEllipsoidExtractor->OpenVisualization(mpMap);

        // Open symmetry
        if(Config::Get<int>("EllipsoidExtraction.Symmetry.Open") == 1)
            mpEllipsoidExtractor->OpenSymmetry();

        std::cout << std::endl;
        cout << " * Open Single-Frame Ellipsoid Estimation. " << std::endl;
        std::cout << std::endl;
    }

    bool Tracking::isKeyFrameForVisualization()
    {
        static Frame* lastVisualizedFrame;
        if( mvpFrames.size() < 2 ) 
        {
            lastVisualizedFrame = mCurrFrame;
            return true;  
        }

        auto lastPose = lastVisualizedFrame->cam_pose_Twc;
        auto currPose = mCurrFrame->cam_pose_Twc;
        auto diffPose = lastPose.inverse() * currPose;
        Vector6d vec = diffPose.toXYZPRYVector();

        if( (vec.head(3).norm() > 0.4) || (vec.tail(3).norm() > M_PI/180.0*15) )  // Visualization param for camera poses
        {
            lastVisualizedFrame = mCurrFrame;
            return true;
        }
        else
            return false;
    }

    void Tracking::OpenOptimization(){
        mbOpenOptimization = true;
        std::cout << std::endl << "Optimization Opens." <<  std::endl << std::endl ;
    }

    void Tracking::CloseOptimization(){
        mbOpenOptimization = false;
        std::cout << std::endl << "Optimization Closes." <<  std::endl << std::endl ;
    }

    void Tracking::OpenGroundPlaneEstimation(){
        miGroundPlaneState = 1;
        pPlaneExtractor = new PlaneExtractor;
        PlaneExtractorParam param;
        param.fx = mK.at<float>(0,0);
        param.fy = mK.at<float>(1,1);
        param.cx = mK.at<float>(0,2);
        param.cy = mK.at<float>(1,2);
        param.scale = Config::Get<double>("Camera.scale");
        pPlaneExtractor->SetParam(param);

        std::cout << " * Open Groundplane Estimation" << std::endl;
        std::cout << std::endl;
    }

    void Tracking::CloseGroundPlaneEstimation(){
        miGroundPlaneState = 0;
        std::cout << std::endl;
        std::cout << " * Close Groundplane Estimation* " << std::endl;
        std::cout << std::endl;
    }

    int Tracking::GetGroundPlaneEstimationState(){
        return miGroundPlaneState;
    }

    void Tracking::ProcessGroundPlaneEstimation()
    {
        cv::Mat depth = mCurrFrame->frame_img;
        g2o::plane groundPlane;
        bool result = pPlaneExtractor->extractGroundPlane(depth, groundPlane);
        if( result )
        {
            g2o::SE3Quat& Twc = mCurrFrame->cam_pose_Twc;  
            groundPlane.transform(Twc);   // transform to the world coordinate.
            mGroundPlane = groundPlane;

            miGroundPlaneState = 2; 

            std::cout << " * Estimate Ground Plane Succeeds: " << mGroundPlane.param.transpose() << std::endl;

            mGroundPlane.color = Vector3d(0.0,0.8,0.0); 
            mpMap->addPlane(&mGroundPlane);

            // Visualize the pointcloud during ground plane extraction
            PointCloudPCL::Ptr pCloud = pPlaneExtractor->GetCloudDense();
            EllipsoidSLAM::PointCloud cloudQuadr = pclToQuadricPointCloud(pCloud);
            EllipsoidSLAM::PointCloud* pCloudQuadr = new EllipsoidSLAM::PointCloud(cloudQuadr);
            EllipsoidSLAM::PointCloud* pCloudQuadrGlobal = transformPointCloud(pCloudQuadr, &mCurrFrame->cam_pose_Twc);
            SetPointCloudProperty(pCloudQuadrGlobal, 0,255,100,2);
            mpMap->clearPointCloud();

            auto vPotentialGroundplanePoints = pPlaneExtractor->GetPotentialGroundPlanePoints();

            srand(time(0));
            for( auto& cloud : vPotentialGroundplanePoints )
            {
                PointCloudPCL::Ptr pCloudPCL(new PointCloudPCL(cloud));
                EllipsoidSLAM::PointCloud cloudQuadri = pclToQuadricPointCloud(pCloudPCL);
                EllipsoidSLAM::PointCloud* pCloudGlobal = transformPointCloud(&cloudQuadri, &Twc);
                
                int r = rand()%155;
                int g = 155;
                int b = rand()%155;
                SetPointCloudProperty(pCloudGlobal, r, g, b, 4);
                mpMap->AddPointCloudList(string("pPlaneExtractor.PotentialGround"), pCloudGlobal, 1);
            }

            // Active the mannual check of groundplane estimation.
            int active_mannual_groundplane_check = Config::Get<int>("Plane.MannualCheck.Open");
            int key = -1;
            bool open_mannual_check = active_mannual_groundplane_check==1;
            bool result_mannual_check = false;
            if(open_mannual_check)
            {
                std::cout << "Estimate Groundplane Done." << std::endl;
                std::cout << "As Groundplane estimation is a simple implementation, please mannually check its correctness." << std::endl;
                std::cout << "Enter Key \'Y\' to confirm, and any other key to cancel this estimation: " << std::endl;

                key = getchar();
            }

            result_mannual_check = (key == 'Y' || key == 'y');            

            if( !open_mannual_check || (open_mannual_check &&  result_mannual_check) )
            {
                // Set groundplane to EllipsoidExtractor
                if( mbDepthEllipsoidOpened ){
                    std::cout << " * Add supporting plane to Ellipsoid Extractor." << std::endl;
                    mpEllipsoidExtractor->SetSupportingPlane(&mGroundPlane);
                }

                // Set groundplane to Optimizer
                std::cout << " * Set groundplane param to optimizer. " << std::endl;
                mpOptimizer->SetGroundPlane(mGroundPlane.param);
            }
            else
            {
                std::cout << " * Cancel this Estimation. " << std::endl;
                miGroundPlaneState = 1;
            }

        }
        else
        {
            std::cout << " * Estimate Ground Plane Fails " << std::endl;
        }


    }

    bool Tracking::SavePointCloudMap(const string& path)
    {
        std::cout << "Save pointcloud Map to : " << path << std::endl;
        mpBuilder->saveMap(path);

        return true;
    }

    // This function saves the object history, which stores all the optimized object vector after every new observations.
    void Tracking::RefreshObjectHistory()
    {
        // Object Vector[11]:  optimized_time[1] | Valid/inValid(1/0)[1] | minimal_vec[9] 
        std::map<int, ellipsoid*> pEllipsoidsMapWithInstance = mpMap->GetAllEllipsoidsMap();
        for( auto pairInsPEllipsoid : pEllipsoidsMapWithInstance )
        {
            int instance = pairInsPEllipsoid.first;
            if( mmObjectHistory.find(instance) == mmObjectHistory.end() )  // when the instance has no record in the history
            {
                MatrixXd obHistory; obHistory.resize(0, 11);
                mmObjectHistory.insert(make_pair(instance, obHistory));
            }

            // Add new history
            VectorXd hisVec; hisVec.resize(11);
            assert(mmObjectObservations.find(instance)!=mmObjectObservations.end() && "How does the ellipsoid get into the map without observations?");

            int currentObs = mmObjectObservations[instance].size();
            hisVec[0] = currentObs;  // observation num.
            hisVec[1] = 1;

            Vector9d vec = pairInsPEllipsoid.second->toMinimalVector(); 
            hisVec.tail<9>() = vec;
            
            // Get the observation num of the last history, add new row if the current observation num is newer.
            MatrixXd &obHisMat = mmObjectHistory[instance];
            if( obHisMat.rows() == 0)
            {   
                // Save to the matrix
                addVecToMatirx(obHisMat, hisVec);
            }
            else {
                int lastObNum = round(obHisMat.row(obHisMat.rows()-1)[0]);
                if( lastObNum == currentObs )       // Compare with last observation
                {
                    // Cover it and remain the same
                    obHisMat.row(obHisMat.rows()-1) = hisVec;
                }
                else
                    addVecToMatirx(obHisMat, hisVec);   // Add a new row.
            }
        }
    }

    // Save the object history into a text file.
    void Tracking::SaveObjectHistory(const string& path)
    {
        /*
        *   TotalInstanceNum
        *   instanceID1 historyNum
        *   0 Valid(1/0) minimalVec
        *   1 Valid(1/0) minimalVec
        *   2 Valid(1/0) minimalVec
        *   ...
        *   instanceID2 historyNum
        *   0 Valid(1/0) minimalVec
        *   1 Valid(1/0) minimalVec
        *   ...
        *   
        */ 
        ofstream out(path.c_str());
        int total_num = mmObjectHistory.size();

        out << total_num << std::endl;
        for( auto obPair : mmObjectHistory )
        {
            int instance = obPair.first;
            MatrixXd &hisMat = obPair.second;

            int hisNum = hisMat.rows();
            out << instance << " " << hisNum << std::endl;
            for( int n=0;n<hisNum; n++)
            {
                VectorXd vec = hisMat.row(n);
                int vecNum = vec.rows();
                for( int i=0; i<vecNum; i++){
                    out << vec[i];
                    if(i==vecNum-1)
                        out << std::endl;
                    else
                        out << " ";
                }
            }
        }
        out.close();
        std::cout << "Save object history to " << path << std::endl;
    }

}