#include "include/core/MapDrawer.h"

#include <pangolin/pangolin.h>
#include <mutex>

#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>

namespace EllipsoidSLAM
{
    // draw axis for ellipsoids
    void drawAxisNormal()
    {
        float length = 2.0;
        
        // x
        glColor3f(1.0,0.0,0.0); // red x
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f, 0.0f);
        glVertex3f(length, 0.0f, 0.0f);
        glEnd();
    
        // y 
        glColor3f(0.0,1.0,0.0); // green y
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f, 0.0f);
        glVertex3f(0.0, length, 0.0f);
    
        glEnd();
    
        // z 
        glColor3f(0.0,0.0,1.0); // blue z
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f ,0.0f );
        glVertex3f(0.0, 0.0f ,length );
    
        glEnd();
    }

    MapDrawer::MapDrawer(const string &strSettingPath, Map* pMap):mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        mCalib << fx,  0,  cx,
                0,  fy, cy,
                0,      0,     1;
    }

    // draw external cubes.
    bool MapDrawer::drawObjects() {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoids();

        std::vector<ellipsoid*> ellipsoidsVisual = mpMap->GetAllEllipsoidsVisual();
        ellipsoids.insert(ellipsoids.end(), ellipsoidsVisual.begin(), ellipsoidsVisual.end());

        for( size_t i=0; i<ellipsoids.size(); i++)
        {
            Eigen::Matrix3Xd corners = ellipsoids[i]->compute3D_BoxCorner();

            glPushMatrix();

            glLineWidth(mCameraLineWidth);

            if(ellipsoids[i]->isColorSet()){
                Vector3d color = ellipsoids[i]->getColor();
                glColor3f(1.0f,0.0f,0.0f);  // red color.
            }
            else
                glColor3f(0.0f,0.0f,1.0f);
            
            glBegin(GL_LINES);

            // draw cube lines. 
            for(int m=0;m<corners.cols();m++){
                for( int n=m+1; n<corners.cols();n++)
                {
                    int m_first = m;
                    glVertex3f(corners(0,m_first),corners(1,m_first),corners(2,m_first));
                    int m_next=n;
                    glVertex3f(corners(0,m_next),corners(1,m_next),corners(2,m_next));
                }
            }
            glEnd();
            glPopMatrix();
        }

        return true;
    }

    // draw ellipsoids
    bool MapDrawer::drawEllipsoids() {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoids();

        std::vector<ellipsoid*> ellipsoidsVisual = mpMap->GetAllEllipsoidsVisual();
        ellipsoids.insert(ellipsoids.end(), ellipsoidsVisual.begin(), ellipsoidsVisual.end());
        for( size_t i=0; i<ellipsoids.size(); i++)
        {
            SE3Quat TmwSE3 = ellipsoids[i]->pose.inverse();
            Vector3d scale = ellipsoids[i]->scale;

            glPushMatrix();

            glLineWidth(mCameraLineWidth*3/4.0);

            if(ellipsoids[i]->isColorSet()){
                Vector4d color = ellipsoids[i]->getColorWithAlpha();
                glColor4f(color(0),color(1),color(2),color(3));
            }
            else
                glColor3f(0.0f,0.0f,1.0f);

            GLUquadricObj *pObj;
            pObj = gluNewQuadric();
            gluQuadricDrawStyle(pObj, GLU_LINE);

            pangolin::OpenGlMatrix Twm;   // model to world
            SE3ToOpenGLCameraMatrix(TmwSE3, Twm);
            glMultMatrixd(Twm.m);  
            glScaled(scale[0],scale[1],scale[2]);

            gluSphere(pObj, 1.0, 26, 13); // draw a sphere with radius 1.0, center (0,0,0), slices 26, and stacks 13.
            drawAxisNormal();

            glPopMatrix();
        }
    }

    // draw all the planes
    bool MapDrawer::drawPlanes() {
        std::vector<plane*> planes = mpMap->GetAllPlanes();
        for( size_t i=0; i<planes.size(); i++) {
            drawPlaneWithEquation(planes[i]);
        }
    }

    // draw a single plane
    void MapDrawer::drawPlaneWithEquation(plane *p) {
        if( p == NULL ) return;

        double pieces = 300;
        double ending, starting;

        // sample x and y
        std::vector<double> x,y,z;
        if( p->mbLimited )      // draw a finite plane.
        {
            pieces = 100;
            
            double area_range = p->mdPlaneSize;
            double step = area_range/pieces;

            // ----- x
            x.clear();
            x.reserve(pieces+2);

            starting = p->mvPlaneCenter[0] - area_range/2;
            ending = p->mvPlaneCenter[0] + area_range/2;

            while(starting <= ending) {
                x.push_back(starting);
                starting += step;
            }

            // ----- y
            y.clear();
            y.reserve(pieces+2);

            starting = p->mvPlaneCenter[1] - area_range/2;
            ending = p->mvPlaneCenter[1] + area_range/2;

            while(starting <= ending) {
                y.push_back(starting);
                starting += step;
            }

            // ----- z
            z.clear();
            z.reserve(pieces+2);

            starting = p->mvPlaneCenter[2] - area_range/2;
            ending = p->mvPlaneCenter[2] + area_range/2;

            while(starting <= ending) {
                z.push_back(starting);
                starting += step;
            }            
        }
        else    // draw an infinite plane, make it big enough
        {
            starting = -5;
            ending = 5;

            x.clear();
            double step = (ending-starting)/pieces;
            x.reserve(pieces+2);
            while(starting <= ending) {
                x.push_back(starting);
                starting += step;
            }
            y=x;
            z=x;
        }
        
        Vector4d param = p->param;
        Vector3d color = p->color;

        glPushMatrix();
        glBegin(GL_POINTS);
        glColor3f(color[0], color[1], color[2]);

        double param_abs_x = std::abs(param[0]);
        double param_abs_y = std::abs(param[1]);
        double param_abs_z = std::abs(param[2]);

        if( param_abs_z > param_abs_x && param_abs_z > param_abs_y ){
            // if the plane is extending toward x axis, use x,y to calculate z.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   Z = (-D-BY-AX)/C
                    double z_  = (-param[3]-param[1]*y[j]-param[0]*x[i])/param[2];

                    glVertex3f(float(x[i]), float(y[j]), float(z_));
                }
            }
        }
        else if( param_abs_x > param_abs_z && param_abs_x > param_abs_y )
        {
            // if the plane is extending toward z axis, use y,z to calculate x.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   X = (-D-BY-CZ)/A
                    double x_  = (-param[3]-param[1]*y[j]-param[2]*z[i])/param[0];

                    glVertex3f(float(x_), float(y[j]), float(z[i]));
                }
            }
        }
        else
        {
            // if the plane is extending toward y axis, use x,z to calculate y.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   Y = (-D-AX-CZ)/B
                    double y_  = (-param[3]-param[0]*x[j]-param[2]*z[i])/param[1];

                    glVertex3f(float(x[j]), float(y_), float(z[i]));
                }
            }
        }
        
        glEnd();
        glPopMatrix();
    }

    bool MapDrawer::drawCameraState() {
        g2o::SE3Quat* cameraState = mpMap->getCameraState();        // Twc
        pangolin::OpenGlMatrix Twc;
        if( cameraState!=NULL )
            SE3ToOpenGLCameraMatrixOrigin(*cameraState, Twc);
        else
        {
            std::cerr << "Can't load camera state." << std::endl;
            Twc.SetIdentity();
        }

        const float &w = mCameraSize*1.5;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m); 
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(1.0f,0.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();

        return true;
    }

    bool MapDrawer::drawTrajectory() {
        std::vector<g2o::SE3Quat*> traj = mpMap->getCameraStateTrajectory();
        for(int i=0; i<traj.size(); i++)
        {
            g2o::SE3Quat* cameraState = traj[i];        // Twc
            pangolin::OpenGlMatrix Twc;

            if( cameraState!=NULL )
                SE3ToOpenGLCameraMatrixOrigin(*cameraState, Twc);
            else
            {
                std::cerr << "Can't load camera state." << std::endl;
                Twc.SetIdentity();
            }

            const float &w = mCameraSize;
            const float h = w*0.75;
            const float z = w*0.6;

            glPushMatrix();

        #ifdef HAVE_GLES
            glMultMatrixf(Twc.m);  
        #else
            glMultMatrixd(Twc.m);
        #endif

            glLineWidth(mCameraLineWidth);
            glColor3f(0.0f,1.0f,0.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }

        return true;
    }

    bool MapDrawer::drawPoints() {
        vector<PointXYZRGB*> pPoints = mpMap->GetAllPoints();
        glPushMatrix();

        for(int i=0; i<pPoints.size(); i=i+1)
        {
            PointXYZRGB &p = *(pPoints[i]);
            glPointSize( p.size );
            glBegin(GL_POINTS);
            glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
            glVertex3d(p.x, p.y, p.z);
            glEnd();
        }

        glPointSize( 1 );
        glPopMatrix();
    }

    // In : Tcw
    // Out: Twc
    void MapDrawer::SE3ToOpenGLCameraMatrix(g2o::SE3Quat &matInSe3, pangolin::OpenGlMatrix &M)
    {
        // eigen to cv
        Eigen::Matrix4d matEigen = matInSe3.to_homogeneous_matrix();
        cv::Mat matIn;
        eigen2cv(matEigen, matIn);

        if(!matIn.empty())
        {
            cv::Mat Rwc(3,3,CV_64F);
            cv::Mat twc(3,1,CV_64F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = matIn.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*matIn.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<double>(0,0);
            M.m[1] = Rwc.at<double>(1,0);
            M.m[2] = Rwc.at<double>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<double>(0,1);
            M.m[5] = Rwc.at<double>(1,1);
            M.m[6] = Rwc.at<double>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<double>(0,2);
            M.m[9] = Rwc.at<double>(1,2);
            M.m[10] = Rwc.at<double>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<double>(0);
            M.m[13] = twc.at<double>(1);
            M.m[14] = twc.at<double>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }

    // not inverse, keep origin
    void MapDrawer::SE3ToOpenGLCameraMatrixOrigin(g2o::SE3Quat &matInSe3, pangolin::OpenGlMatrix &M)
    {
        // eigen to cv
        Eigen::Matrix4d matEigen = matInSe3.to_homogeneous_matrix();
        cv::Mat matIn;
        eigen2cv(matEigen, matIn);

        if(!matIn.empty())
        {
            cv::Mat Rwc(3,3,CV_64F);
            cv::Mat twc(3,1,CV_64F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = matIn.rowRange(0,3).colRange(0,3);
                twc = matIn.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<double>(0,0);
            M.m[1] = Rwc.at<double>(1,0);
            M.m[2] = Rwc.at<double>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<double>(0,1);
            M.m[5] = Rwc.at<double>(1,1);
            M.m[6] = Rwc.at<double>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<double>(0,2);
            M.m[9] = Rwc.at<double>(1,2);
            M.m[10] = Rwc.at<double>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<double>(0);
            M.m[13] = twc.at<double>(1);
            M.m[14] = twc.at<double>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }

    void MapDrawer::setCalib(Eigen::Matrix3d& calib)
    {
        mCalib = calib;
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
        g2o::SE3Quat *cameraState = mpMap->getCameraState();        // Twc

        if (cameraState != NULL)
            SE3ToOpenGLCameraMatrixOrigin(*cameraState, M);
        else {
            M.SetIdentity();
        }
    }

    void MapDrawer::drawPointCloudLists()
    {
        auto pointLists = mpMap->GetPointCloudList();

        glPushMatrix();

        for(auto pair:pointLists){
            auto pPoints = pair.second;
            if( pPoints == NULL ) continue;
            for(int i=0; i<pPoints->size(); i=i+1)
            {
                PointXYZRGB &p = (*pPoints)[i];
                glPointSize( p.size );
                glBegin(GL_POINTS);
                glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
                glVertex3d(p.x, p.y, p.z);
                glEnd();

            }
        }
        glPointSize( 1 );

        glPopMatrix();
    }

    void MapDrawer::drawPointCloudWithOptions(const std::map<std::string,bool> &options)
    {
        auto pointLists = mpMap->GetPointCloudList();
        glPushMatrix();

        for(auto pair:pointLists){
            auto pPoints = pair.second;
            if( pPoints == NULL ) continue;

            auto iter = options.find(pair.first);
            if(iter == options.end()) {
                continue;  // not exist
            }
            if(iter->second == false) continue; // menu is closed

            for(int i=0; i<pPoints->size(); i=i+1)
            {
                PointXYZRGB &p = (*pPoints)[i];
                glPointSize( p.size );
                glBegin(GL_POINTS);
                glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
                glVertex3d(p.x, p.y, p.z);
                glEnd();

            }
        }
        glPointSize( 1 );
        glPopMatrix();        
    }


}
