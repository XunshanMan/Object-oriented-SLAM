#include "PlaneExtractor.h"

#include "src/symmetry/PointCloudFilter.h"

#include <src/config/Config.h>

namespace EllipsoidSLAM
{
    bool compare_func_plane_dis(g2o::plane* &p1, g2o::plane* &p2)
    {
        Vector3d cam_pose(0,0,0);
        double dis1 = p1->distanceToPoint(cam_pose);        
        double dis2 = p2->distanceToPoint(cam_pose);        
        return dis1 < dis2;
    }

    bool compare_func_plane_size(std::pair<g2o::plane*, int> &p1, std::pair<g2o::plane*, int> &p2)
    {
        return p1.second > p2.second;
    }

    void PlaneExtractor::extractPlanes(const cv::Mat &imDepth) {
        mvPlaneCoefficients.clear();
        mvPlanePoints.clear();

        int row_start;
        if( mParam.RangeOpen )  // if this flag is opened, only the bottom half part of the depth image is considered
            row_start = imDepth.rows - mParam.RangeHeight;  // it normally starts from the half height of the image 
        else
            row_start = 0;
        PointCloudPCL::Ptr inputCloud( new PointCloudPCL() );
        for ( int m=row_start; m<imDepth.rows; m+=1 )
        {
            for ( int n=0; n<imDepth.cols; n+=1 )
            {
                ushort depthValue = imDepth.ptr<ushort>(m)[n];
                
                double d = depthValue/ mParam.scale;
                PointT p;
                p.z = d;
                p.x = ( n - mParam.cx) * p.z / mParam.fx;
                p.y = ( m - mParam.cy) * p.z / mParam.fy;
                p.r = 0;
                p.g = 0;
                p.b = 250;

                inputCloud->points.push_back(p);
            }
        }
        inputCloud->height = ceil(imDepth.rows-row_start);
        inputCloud->width = ceil(imDepth.cols);

        mpCloudDense = inputCloud;

        // estimate the normal
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.05f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setInputCloud(inputCloud);
        ne.compute(*cloud_normals);

        // load config parameters
        int min_plane = Config::Get<int>("Plane.MinSize");
        float AngTh = Config::Get<float>("Plane.AngleThreshold");
        float DisTh = Config::Get<float>("Plane.DistanceThreshold");

        vector<pcl::ModelCoefficients> coefficients;
        vector<pcl::PointIndices> inliers;
        pcl::PointCloud<pcl::Label>::Ptr labels ( new pcl::PointCloud<pcl::Label> );
        vector<pcl::PointIndices> label_indices;
        vector<pcl::PointIndices> boundary;

        pcl::OrganizedMultiPlaneSegmentation< PointT, pcl::Normal, pcl::Label > mps;
        mps.setMinInliers (100);
        mps.setAngularThreshold (0.017453 * AngTh);  // deg to rad
        mps.setDistanceThreshold (DisTh);
        mps.setInputNormals (cloud_normals);
        mps.setInputCloud (inputCloud);
        std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>> regions;
        mps.segmentAndRefine (regions, coefficients, inliers, labels, label_indices, boundary);

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(inputCloud);
        extract.setNegative(false);

        // save the points belonging to each planes, and the coefficients
        for (int i = 0; i < inliers.size(); ++i) {
            int planeSize = inliers[i].indices.size();
            if( planeSize < min_plane ) continue;

            PointCloudPCL::Ptr planeCloud(new PointCloudPCL());
            cv::Mat coef = (cv::Mat_<float>(4,1) << coefficients[i].values[0],
                    coefficients[i].values[1],
                    coefficients[i].values[2],
                    coefficients[i].values[3]);
            if(coef.at<float>(3) < 0)
                coef = -coef;

            extract.setIndices(boost::make_shared<pcl::PointIndices>(inliers[i]));
            extract.filter(*planeCloud);

            mvPlanePoints.push_back(*planeCloud);
            mvPlaneCoefficients.push_back(coef);        
        }
    }

    void PlaneExtractor::SetParam(PlaneExtractorParam& param){
        mParam = param;
    }


    std::vector<PointCloudPCL> PlaneExtractor::GetPoints(){
        return mvPlanePoints;
    }

    std::vector<PointCloudPCL> PlaneExtractor::GetPotentialGroundPlanePoints(){
        return mvPotentialGroundPlanePoints;
    }

    std::vector<cv::Mat> PlaneExtractor::GetCoefficients(){
        return mvPlaneCoefficients;
    }

    bool PlaneExtractor::extractGroundPlane(const cv::Mat &depth, g2o::plane& plane){
        mParam.RangeOpen = false;
        mParam.RangeHeight = depth.rows/2;

        extractPlanes(depth);   // first extract all the potential planes
        
        if( mvPlaneCoefficients.size() < 1)     // there should be more than 1 potential planes 
            return false;

        // the groundplane should meet two criteria:
        // 1) its normal vector should have a small angle difference with the gravity direction. ( filter the walls )
        // 2) its distance with the center of the camera is the smallest among all the extracted planes.

        Vector3d cam_pose(0,0,0);   
        std::vector<g2o::plane*> vpPlanes;
        std::vector<std::pair<g2o::plane*, int>> mapPlaneSize;
        for( int i=0; i<mvPlaneCoefficients.size(); i++ )
        {
            Eigen::Vector4d vec;
            auto& coeff = mvPlaneCoefficients[i];
            vec << double(coeff.at<float>(0)), double(coeff.at<float>(1)), double(coeff.at<float>(2)), double(coeff.at<float>(3));

            // suppose the Y axis of the camera coordinate is the gravity direction and set a tolerence of angle difference of [pi/4, 3*pi/4].
            // it always meets the criteria when the camera is located on a mobile robot.
            // it is a loose criteria only for filtering the walls.
            Eigen::Vector3d axisY(0,1,0);       
            Eigen::Vector3d axisNorm = vec.head(3);
            double cos_theta = axisNorm.transpose() * axisY;
            cos_theta = cos_theta / axisNorm.norm() / axisY.norm();

            double theta = acos( cos_theta );      // acos : [0,pi]
            if( theta > M_PI/4 && theta < 3*M_PI/4 ) continue;  

            g2o::plane* pPlane = new g2o::plane();
            pPlane->param= vec;
            vpPlanes.push_back(pPlane);
            mvPotentialGroundPlanePoints.push_back(mvPlanePoints[i]);
            mapPlaneSize.push_back(make_pair(pPlane, mvPlanePoints[i].size()));
        }

        if( vpPlanes.size() < 1) {      // there should be more than 1 valid planes 
            std::cout << "Please let the camera be parallel to the ground for initialization." << std::endl;
            return false;
        }

        // sort according to the size of planes
        sort(mapPlaneSize.begin(), mapPlaneSize.end(), compare_func_plane_size);
        g2o::plane* pPlaneGround = mapPlaneSize[0].first;

        // adjust the flag of the plane coefficients to make the distance between the camera center and the plane positive
        double value = pPlaneGround->distanceToPoint(Vector3d(0,0,0), true);
        if( value < 0 ) 
            pPlaneGround->param = -pPlaneGround->param;

        plane = *pPlaneGround;

        return true;
    }

    PlaneExtractor::PlaneExtractor(const string& settings){
        std::cout << "Init plane extractor using : " << settings << std::endl;  
        Config::Init();
        Config::SetParameterFile(settings);
        mParam.fx = Config::Get<double>("Camera.fx"); 
        mParam.fy = Config::Get<double>("Camera.fy"); ; 
        mParam.cx = Config::Get<double>("Camera.cx"); ; 
        mParam.cy = Config::Get<double>("Camera.cy"); ; 
        mParam.scale = Config::Get<double>("DepthMapFactor"); ; 
    }

    PointCloudPCL::Ptr PlaneExtractor::GetCloudDense(){
        return mpCloudDense;
    }
}