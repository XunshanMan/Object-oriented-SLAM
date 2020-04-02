#include "PointCloudFilter.h"

Vector2d getXYCenterOfPointCloud(EllipsoidSLAM::PointCloud* pPoints)
{
    double x=0;
    double y=0;
    int num=0;
    for( auto p:(*pPoints))
    {
        x+=p.x;
        y+=p.y;
        num++;
    }
    x = x / double(num);
    y = y / double(num);
    return Vector2d(x,y);
}


// coordinates : x -> right, y-> down, z->front 
EllipsoidSLAM::PointCloud getPointCloudInRect(cv::Mat &depth, cv::Mat &rgb, const VectorXd &detect, EllipsoidSLAM::camera_intrinsic &camera, double range) {
    // detect : x1 y1 x2 y2 
    EllipsoidSLAM::PointCloud cloud;

    // scan the points in the bounding box 
    int x1 = int(detect(0));
    int y1 = int(detect(1));
    int x2 = int(detect(2));
    int y2 = int(detect(3));

    for (int y = y1; y < y2; y = y+3){
        for (int x = x1; x < x2; x = x+3) {
            ushort *ptd = depth.ptr<ushort>(y);
            ushort d = ptd[x];

            EllipsoidSLAM::PointXYZRGB p;
            p.z = d / camera.scale;
            if (p.z <= 0.1 || p.z > range)    // if the depth is valid
                continue;

            p.x = (x - camera.cx) * p.z / camera.fx;
            p.y = (y - camera.cy) * p.z / camera.fy;

            p.b = rgb.ptr<uchar>(y)[x * 3];
            p.g = rgb.ptr<uchar>(y)[x * 3 + 1];
            p.r = rgb.ptr<uchar>(y)[x * 3 + 2];

            p.size = 1;

            cloud.push_back(p);
        }
    }

    return cloud;

}

EllipsoidSLAM::PointCloud getPointCloudInRect(cv::Mat &depth, const VectorXd &detect, EllipsoidSLAM::camera_intrinsic &camera, double range) {
    cv::Mat rgb = cv::Mat(depth.rows, depth.cols, CV_8UC3, cv::Scalar(0,0,0));    
    return getPointCloudInRect(depth, rgb, detect, camera, range);
}

void filterGround(EllipsoidSLAM::PointCloud** ppCloud)
{
    EllipsoidSLAM::PointCloud *pCloudFiltered = new EllipsoidSLAM::PointCloud;
    EllipsoidSLAM::PointCloud *pCloud = *ppCloud;
    int num = pCloud->size();
    for(auto p: (*pCloud))
    {
        if(p.z > 0.05)
            pCloudFiltered->push_back(p);
    }

    delete pCloud;
    (*ppCloud) = pCloudFiltered;
}

void outputCloud(EllipsoidSLAM::PointCloud *pCloud, int num )
{
    int total_num = pCloud->size();

    cout << "===== Point Cloud ====" << endl;
    int count = 0;
    for(auto p: *pCloud)
    {
        cout << count << ": " << p.x <<","<< p.y <<","<< p.z << endl ;
        if(count++ > num)
            break;
    }
}

EllipsoidSLAM::PointCloud pclToQuadricPointCloud(PointCloudPCL::Ptr &pCloud)
{
    EllipsoidSLAM::PointCloud cloud;
    int num = pCloud->points.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB p;
        PointT pT = pCloud->points[i];
        p.r = pT.r;
        p.g = pT.g;
        p.b = pT.b;

        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        cloud.push_back(p);
    }

    return cloud;
}

EllipsoidSLAM::PointCloud* pclToQuadricPointCloudPtr(PointCloudPCL::Ptr &pCloud)
{
    EllipsoidSLAM::PointCloud* cloudPtr = new EllipsoidSLAM::PointCloud;
    EllipsoidSLAM::PointCloud& cloud = *cloudPtr;
    int num = pCloud->points.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB p;
        PointT pT = pCloud->points[i];
        p.r = pT.r;
        p.g = pT.g;
        p.b = pT.b;

        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        cloud.push_back(p);
    }

    return cloudPtr;
}

PointCloudPCL::Ptr QuadricPointCloudToPcl(EllipsoidSLAM::PointCloud &cloud)
{
    PointCloudPCL::Ptr pCloud(new PointCloudPCL);
    
    int num = cloud.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB pT = cloud[i];
        
        PointT p;
        p.r = pT.r;
        p.g = pT.g;
        p.b = pT.b;

        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        pCloud->points.push_back(p);
    }

    return pCloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr QuadricPointCloudToPclXYZ(EllipsoidSLAM::PointCloud &cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    int num = cloud.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB pT = cloud[i];
        
        pcl::PointXYZ p;
        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        pCloud->points.push_back(p);
    }
    pCloud->width = (int) pCloud->points.size ();  
    pCloud->height = 1;

    return pCloud;
}

EllipsoidSLAM::PointCloud pclXYZToQuadricPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &pCloud)
{
    EllipsoidSLAM::PointCloud cloud;
    int num = pCloud->points.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB p;
        pcl::PointXYZ pT = pCloud->points[i];
        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        cloud.push_back(p);
    }

    return cloud;
}

EllipsoidSLAM::PointCloud* pclXYZToQuadricPointCloudPtr(pcl::PointCloud<pcl::PointXYZ>::Ptr &pCloud)
{
    EllipsoidSLAM::PointCloud* cloudPtr = new EllipsoidSLAM::PointCloud;
    EllipsoidSLAM::PointCloud &cloud = *cloudPtr;
    int num = pCloud->points.size();
    for(int i=0;i<num;i++){
        EllipsoidSLAM::PointXYZRGB p;
        pcl::PointXYZ pT = pCloud->points[i];
        p.x = pT.x;
        p.y = pT.y;
        p.z = pT.z;
        cloud.push_back(p);
    }

    return cloudPtr;
}


void 
DownSamplePointCloud(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, int param_num)
{
    clock_t startTime,endTime;
    startTime = clock();
    PointCloudPCL::Ptr pPclCloud = QuadricPointCloudToPcl(cloudIn);
    endTime = clock();

    clock_t startTime_downsample = clock();

    pcl::VoxelGrid<PointT> voxel;
    double gridsize = 0.02;
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( pPclCloud );
    PointCloudPCL::Ptr tmp( new PointCloudPCL() );
    voxel.filter( *tmp );
    clock_t endTime_downsample = clock();

    clock_t startTime_outlier = clock();
    PointCloudPCL::Ptr pPclCloudFiltered(new PointCloudPCL);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;   
    sor.setInputCloud (tmp);                           
    sor.setMeanK (param_num);                              
    sor.setStddevMulThresh (1.0);                     
    sor.filter (*pPclCloudFiltered);                  

    clock_t endTime_outlier = clock();


    clock_t startTime2,endTime2;
    startTime2 = clock();
    cloudOut = pclToQuadricPointCloud(pPclCloudFiltered);
    endTime2 = clock();

    // cout << "Time diff QuadricPointCloudToPcl: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    // cout << "Time diff downsample: " <<(double)(endTime_downsample - startTime_downsample) / CLOCKS_PER_SEC << "s" << endl;
    // cout << "Time diff outlier: " <<(double)(endTime_outlier - startTime_outlier) / CLOCKS_PER_SEC << "s" << endl;
}

void DownSamplePointCloudOnly(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, double grid)
{
    PointCloudPCL::Ptr pPclCloud = QuadricPointCloudToPcl(cloudIn);

    pcl::VoxelGrid<PointT> voxel;
    double gridsize = grid; 
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( pPclCloud );
    PointCloudPCL::Ptr tmp( new PointCloudPCL() );
    voxel.filter( *tmp );

    cloudOut = pclToQuadricPointCloud(tmp);
}

void FiltOutliers(EllipsoidSLAM::PointCloud& cloudIn, EllipsoidSLAM::PointCloud& cloudOut, int num_neighbor)
{
    PointCloudPCL::Ptr pPclCloud = QuadricPointCloudToPcl(cloudIn);

    PointCloudPCL::Ptr pPclCloudFiltered(new PointCloudPCL);
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;   
    sor.setInputCloud (pPclCloud);                          
    sor.setMeanK (num_neighbor);                            
    sor.setStddevMulThresh (1.0);                    
    sor.filter (*pPclCloudFiltered);                 

    cloudOut = pclToQuadricPointCloud(pPclCloudFiltered);
}

void FiltPointsInBox(EllipsoidSLAM::PointCloud* pPoints_global, EllipsoidSLAM::PointCloud* pPoints_global_inBox, g2o::ellipsoid &e){
    double radius = MAX(MAX(e.scale[0], e.scale[1]), e.scale[2]);
    Vector3d center = e.toVector().head(3);
    
    pPoints_global_inBox->clear();

    Eigen::Matrix4d Q_star = e.generateQuadric();
    Eigen::Matrix4d Q = Q_star.inverse();
    for(auto p : (*pPoints_global) )        
    {
        Vector3d xyz(p.x, p.y, p.z);

        Eigen::Vector4d X = real_to_homo_coord_vec<double>(xyz);

        double abstract_dis = X.transpose() * Q * X;
        bool isInside = (abstract_dis <0);

        if( isInside )
            pPoints_global_inBox->push_back(p);
    }

    return;
}

// add the points in point cloud p2 to p1
void CombinePointCloud(EllipsoidSLAM::PointCloud *p1, EllipsoidSLAM::PointCloud *p2){
    if( p1 ==NULL || p2==NULL){
        cerr<< " point cloud is NULL. " << endl;
        return;
    }
    for(auto point : *p2)
        p1->push_back(point);

    return ;    
}