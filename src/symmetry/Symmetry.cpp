#include "PointCloudFilter.h"
#include "SymmetrySolver.h"
#include "Symmetry.h"

#include <ctime>

namespace EllipsoidSLAM
{
    // ---- compare functions
    bool comp_func_mapPlane(const pair<double,g2o::plane*>& t1, const pair<double,g2o::plane*>& t2)
    {  
        return t1.first > t2.first;  
    }  

    bool comp_func_mapProbData(const pair<double,SymmetrySolverData>& t1, const pair<double,SymmetrySolverData>& t2)
    {  
        return t1.first > t2.first;  
    }  

    void SaveSymmetryResultToText(std::vector<pair<double, SymmetrySolverData>> &mapProbData, 
            const string& path)
    {
        ofstream out(path.c_str());

        for( auto pair : mapProbData )
        {
            double prob = pair.first;
            auto data = pair.second;

            out <<
            data.pInitPlane->azimuth() << "\t" <<
            data.pInitPlane->distance() << "\t" <<
            data.init_error << "\t" <<

            data.pPlane->azimuth() << "\t" <<
            data.pPlane->distance() << "\t" <<
            data.final_error << std::endl;

        }

        std::cout << "[ Save symmetry result to : " << path << " ]" << std::endl;
    }

    std::vector<g2o::plane*> GenerateInitPlanes(int symType = 1)
    {
        /*
        *   sample new planes along two degrees: angle and distance.
        *   on each of them sample (2*step_num+1) planes.
        *   totally, sample (step_num*2+1)*(step_num*2+1) planes around the init planes.
        */
        bool open_multiple_samples = true;

        if(open_multiple_samples)
        {
            // CONFIGURATION
            int step_num = 1;
            double diff_dis = 0.2;  // 0.2m
            double diff_angle = M_PI/180.0*5;  // 5deg

            double start_dis = -diff_dis * step_num;
            double start_angle = -diff_angle * step_num;

            int loop_num = 2 * step_num + 1;
            // ------------
            std::vector<g2o::plane*> initPlanes;
            for(int i = 0; i<loop_num; i++){
                for( int m=0;m<loop_num; m++)
                {
                        double dis = start_dis + diff_dis * i;
                        double angle = start_angle + diff_angle * m;
                        g2o::plane *pPlane = new g2o::plane;
                        pPlane->fromDisAngleTrans(dis, angle, 0);
                        initPlanes.push_back(pPlane);
                }
            }
            return initPlanes;
        }
        else
        {
            std::vector<g2o::plane*> initPlanes;
            g2o::plane *pPlane = new g2o::plane;
            pPlane->fromDisAngleTrans(0, 0, 0);
            initPlanes.push_back(pPlane);
            return initPlanes;
        }
    }

    SymmetrySolverData Symmetry::estimateSymmetry(Vector4d &bbox, PointCloud* pCloud, VectorXd& pose, cv::Mat& projDepth, camera_intrinsic& camera, 
        int symType)
    {
        SymmetrySolverData output; output.result = false;
        if( pCloud->size() < 1 ) 
        {
            std::cerr << " Point Cloud Size has some problems:  " << pCloud->size() << std::endl;
            return output;
        }

        Matrix3d calib = getCalibFromCamera(camera);
        
        SymmetrySolver solver;
        solver.SetCameraParam(camera);
        // set the border
        if( mbOpenSparseEstimation )
            solver.SetBorders(mpBorders);
        std::vector<pair<double, SymmetrySolverData>> mapProbData;

        std::vector<g2o::plane*> initPlanes = GenerateInitPlanes(symType);

        int opt_time = initPlanes.size();
        for(int i=0; i< opt_time ; i ++){        
            g2o::plane* pPlane= initPlanes[i];
            SymmetrySolverData data;
            if(symType == 1)
                data = solver.OptimizeSymmetryPlane(bbox, *pPlane, pCloud, projDepth, pose, calib, camera.scale, symType);
            else if(symType == 2)
                data = solver.OptimizeSymmetryDualPlane(bbox, *pPlane, pCloud, projDepth, pose, calib, camera.scale, symType);
            double prob = data.prob;
            mapProbData.push_back(make_pair(prob, data));
        }


        sort(mapProbData.begin(), mapProbData.end(), comp_func_mapProbData);
        assert(mapProbData.size() > 0 && "Wrong size in mapProbData.");
        auto pair = mapProbData[0];

        output = pair.second;
        return output;
    }

    void Symmetry::releaseData(SymmetryOutputData& data)
    {
        if(data.pCloud != NULL)
            delete data.pCloud;
    }

    // change the distance between a point to the image plane to the distance between a point to the camera center
    double calculateProjZ(double f, double d, double xi, double yi){
        return d*sqrt(xi*xi+f*f+yi*yi)/f;
    }

    cv::Mat Symmetry::getProjDepthMat(cv::Mat& depth, camera_intrinsic& camera){
        int rows = depth.rows;
        int cols = depth.cols;

        cv::Mat depthProj = cv::Mat(rows, cols, CV_16UC1, cv::Scalar(0));

        double fx = camera.fx;
        double cx = camera.cx;
        double cy = camera.cy;
        for( int x=0;x<cols;x++ )
            for( int y=0;y<rows;y++ )
            {
                ushort d = depth.at<ushort>(y,x);

                double realz = calculateProjZ(fx, double(d), (x - cx), (y - cy));
                
                depthProj.at<ushort>(y, x)  = ushort(realz);
            }
            
        return depthProj;
    }

    void Symmetry::SetBorders(EllipsoidSLAM::PointCloud* pBorders)
    {
        mbOpenSparseEstimation = true;
        mpBorders = pBorders;
    }

    Symmetry::Symmetry()
    {
        mbOpenSparseEstimation = false;

        mpExtractor = new BorderExtractor;

        miParamFilterPointNum = 100;

        mbGroundPlaneSet = false;

        mbObjInitialGuessSet = false;

        mbInitPlanesSet = false;
    }

}