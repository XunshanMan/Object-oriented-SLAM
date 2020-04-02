#include "include/core/Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

namespace EllipsoidSLAM {

    Viewer::Viewer(System *pSystem, const string &strSettingPath, EllipsoidSLAM::MapDrawer *pMapDrawer){
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];

        mbFinishRequested=false;
        mpSystem = pSystem;
        mpMapDrawer = pMapDrawer;

        miRows = fSettings["Camera.height"];
        miCols = fSettings["Camera.width"];

        mvMenuStruct.clear();
    }

    Viewer::Viewer(const string &strSettingPath, MapDrawer* pMapDrawer):mpMapDrawer(pMapDrawer) {

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];

        mbFinishRequested=false;

        miRows = fSettings["Camera.height"];
        miCols = fSettings["Camera.width"];

        mvMenuStruct.clear();
    }

    void Viewer::SetFrameDrawer(FrameDrawer* pFrameDrawer)
    {
        mpFrameDrawer = pFrameDrawer;
    }

    void Viewer::run() {
        mbFinished = false;

        pangolin::CreateWindowAndBind("EllipsoidSLAM: Map Viewer", 1024, 768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowEllipsoids("menu.Show Ellipsoids", true, true);
        pangolin::Var<bool> menuShowPlanes("menu.Show Planes", true, true);
        pangolin::Var<bool> menuShowCuboids("menu.Show Cuboids", false, true);

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View &d_cam = pangolin::Display("cam")
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -float(miCols) / float(miRows))
                .SetHandler(new pangolin::Handler3D(s_cam));
        
        // Add view for images
        pangolin::View& rgb_image = pangolin::Display("rgb")
        .SetBounds(0,0.3,0.2,0.5,float(miCols) / float(miRows))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);

        pangolin::View& depth_image = pangolin::Display("depth")
        .SetBounds(0,0.3,0.5,0.8,float(miCols) / float(miRows))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        bool bFollow = true;

        pangolin::GlTexture imageTexture(miCols,miRows,GL_RGB,false,0,GL_BGR,GL_UNSIGNED_BYTE);

        while (1) {
            RefreshMenu();  // Deal with dynamic menu bars

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc); // get current camera pose
            
            if(menuFollowCamera && bFollow) // Follow camera
            {
                s_cam.Follow(Twc);
            }
            else if(menuFollowCamera && !bFollow)
            {
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if(!menuFollowCamera && bFollow)
            {
                bFollow = false;
            }

            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);

            pangolin::glDrawAxis(3);    // draw world coordintates

            if(menuShowKeyFrames)
            {
                mpMapDrawer->drawCameraState();
                mpMapDrawer->drawTrajectory();
            }

            // draw external cubes of ellipsoids 
            if(menuShowCuboids)
                mpMapDrawer->drawObjects();

            // draw ellipsoids
            if(menuShowEllipsoids)
                mpMapDrawer->drawEllipsoids();

            // draw planes, including grounplanes and symmetry planes
            if(menuShowPlanes)
                mpMapDrawer->drawPlanes();

            mpMapDrawer->drawPoints();  // draw point clouds

            // draw pointclouds with names
            RefreshPointCloudOptions();
            mpMapDrawer->drawPointCloudWithOptions(mmPointCloudOptionMap);
            // mpMapDrawer->drawPointCloudLists();

            // draw images : rgb
            cv::Mat rgb = mpFrameDrawer->getCurrentFrameImage();
            if(!rgb.empty())
            {
                imageTexture.Upload(rgb.data,GL_BGR,GL_UNSIGNED_BYTE);
                //display the image
                rgb_image.Activate();
                glColor3f(1.0,1.0,1.0);
                imageTexture.RenderToViewportFlipY();
            }

            // draw images : depth
            cv::Mat depth = mpFrameDrawer->getCurrentDepthFrameImage();
            if(!depth.empty())
            {
                imageTexture.Upload(depth.data,GL_BGR,GL_UNSIGNED_BYTE);
                //display the image
                depth_image.Activate();
                glColor3f(1.0,1.0,1.0);
                imageTexture.RenderToViewportFlipY();
            }

            pangolin::FinishFrame();

            if (CheckFinish())
                break;
        }

        SetFinish();
    }


    bool Viewer::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    void Viewer::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }


    bool Viewer::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    int Viewer::addDoubleMenu(string name, double min, double max, double def){
        unique_lock<mutex> lock(mMutexFinish);

        MenuStruct menu;
        menu.min = min;
        menu.max = max;
        menu.def = def;
        menu.name = name;
        mvMenuStruct.push_back(menu);

        return mvMenuStruct.size()-1;
    }

    bool Viewer::getValueDoubleMenu(int id, double &value){
        unique_lock<mutex> lock(mMutexFinish);
        if( 0 <= id && 0< mvDoubleMenus.size())
        {
            value = mvDoubleMenus[id]->Get();
            return true;
        }
        else
        {
            return false;
        }
        
    }

    void Viewer::RefreshPointCloudOptions()
    {
        // generate options from mmPointCloudOptionMenus, pointclouds with names will only be drawn when their options are activated.
        std::map<std::string,bool> options;
        for( auto pair : mmPointCloudOptionMenus)
            options.insert(make_pair(pair.first, pair.second->Get()));
        
        mmPointCloudOptionMap.clear();
        mmPointCloudOptionMap = options;
    }

    void Viewer::RefreshMenu(){
        unique_lock<mutex> lock(mMutexFinish);

        // Generate menu bar for every pointcloud in pointcloud list.
        auto pointLists = mpSystem->getMap()->GetPointCloudList();

        // Iterate over the menu and delete the menu if the corresponding clouds are no longer available
        for( auto menuPair = mmPointCloudOptionMenus.begin(); menuPair!=mmPointCloudOptionMenus.end(); menuPair++)
        {
            if(pointLists.find(menuPair->first) == pointLists.end())
            {
                delete menuPair->second;        // destroy the dynamic menu 
                mmPointCloudOptionMenus.erase(menuPair);  
            }
        }

        // Iterate over the cloud lists to add new menu.
        for( auto cloudPair: pointLists )
        {
            if(mmPointCloudOptionMenus.find(cloudPair.first) == mmPointCloudOptionMenus.end())
            {
                pangolin::Var<bool>* pMenu = new pangolin::Var<bool>(string("menu.") + cloudPair.first, true, true);
                mmPointCloudOptionMenus.insert(make_pair(cloudPair.first, pMenu));            
            }
        }

        // refresh double bars
        int doubleBarNum = mvDoubleMenus.size();
        int structNum = mvMenuStruct.size();
        if( structNum > 0 && structNum > doubleBarNum )
        {
            for(int i = doubleBarNum; i < structNum; i++)
            {
                pangolin::Var<double>* pMenu = new pangolin::Var<double>(string("menu.")+mvMenuStruct[i].name, mvMenuStruct[i].def, mvMenuStruct[i].min, mvMenuStruct[i].max);
                mvDoubleMenus.push_back(pMenu);
            }
        }

    }
}

