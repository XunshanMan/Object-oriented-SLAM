#include "BorderExtractor.h"

namespace EllipsoidSLAM
{
pcl::PointCloud<PointType>::Ptr BorderExtractor::ToPointTCloud(pcl::PointCloud<pcl::PointWithRange>::Ptr pCloud)
{
  pcl::PointCloud<PointType>::Ptr pCloudXYZ(new pcl::PointCloud<PointType>);
  for( auto p : pCloud->points )
  {
    PointType pXYZ;
    pXYZ.x = p.x;
    pXYZ.y = p.y;
    pXYZ.z = p.z;

    pCloudXYZ->points.push_back(pXYZ);
    
  }

  return pCloudXYZ;
}

pcl::PointCloud<PointType>::Ptr BorderExtractor::CombineCloud(pcl::PointCloud<PointType>::Ptr& pCloud1, pcl::PointCloud<PointType>::Ptr& pCloud2, pcl::PointIndices& indices1)
{
  pcl::PointCloud<PointType>::Ptr pComposite(new pcl::PointCloud<PointType>);
  
  int i=0;
  for( auto p: pCloud1->points ){
    pComposite->points.push_back(p);
    indices1.indices.push_back(i++);
  }
  for( auto p: pCloud2->points )
    pComposite->points.push_back(p);
  return pComposite;
}

pcl::PointCloud<PointType>::Ptr 
BorderExtractor::FilterBordersBasedOnPointCloud(pcl::PointCloud<pcl::PointWithRange>::Ptr& pBordersNoisy, pcl::PointCloud<PointType>::Ptr &point_cloud_ptr){

  pcl::PointCloud<PointType>::Ptr pBorders_PointTCloud = ToPointTCloud(pBordersNoisy);

  pcl::PointIndices indices;
  pcl::PointCloud<PointType>::Ptr pComposite = CombineCloud(pBorders_PointTCloud, point_cloud_ptr, indices);

  // filter the outliers of the borders.
  pcl::PointCloud<PointType>::Ptr borderCloudFiltered(new pcl::PointCloud<PointType>());

  pcl::RadiusOutlierRemoval<PointType> outrem;
  outrem.setInputCloud(pComposite);
  outrem.setIndices(boost::make_shared<pcl::PointIndices>(indices));
  outrem.setRadiusSearch(0.05); 
  outrem.setMinNeighborsInRadius(6);
  outrem.filter(*borderCloudFiltered);

  return borderCloudFiltered;

}

pcl::PointCloud<pcl::PointWithRange>::Ptr BorderExtractor::extractBordersFromPointCloud(pcl::PointCloud<PointType>::Ptr& point_cloud_ptr, double resolution)
{
    clock_t startTime = clock();

    float angular_resolution = pcl::deg2rad( float(resolution) );
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    bool setUnseenToMaxRange = true;

    Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());

    pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;

    pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;

    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                            point_cloud.sensor_origin_[1],
                                                            point_cloud.sensor_origin_[2])) *
                    Eigen::Affine3f (point_cloud.sensor_orientation_);
    
  // -----------------------------------------------
  // -----Create RangeImage from the PointCloud-----
  // -----------------------------------------------
  float noise_level = 0.0;
  float min_range = 0.0f;
  int border_size = 1;
  pcl::RangeImage::Ptr range_image_ptr (new pcl::RangeImage);
  pcl::RangeImage& range_image = *range_image_ptr;   
  range_image.createFromPointCloud (point_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                   scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  range_image.integrateFarRanges (far_ranges);

  if (setUnseenToMaxRange)
    range_image.setUnseenToMaxRange ();
  // -------------------------
  // -----Extract borders-----
  // -------------------------
  pcl::RangeImageBorderExtractor border_extractor (&range_image);
  pcl::PointCloud<pcl::BorderDescription> border_descriptions;

  // set parameter
  pcl::RangeImageBorderExtractor::Parameters& param = border_extractor.getParameters();
  // param.pixel_radius_borders =  1; //  default : 3
  // param.minimum_border_probability = 0.99; // defualt: 0.8
  border_extractor.compute (border_descriptions);
  
  // ----------------------------------
  // -----Show points in 3D viewer-----
  // ----------------------------------
  pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
                                            veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
                                            shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
  pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr,
                                      & veil_points = * veil_points_ptr,
                                      & shadow_points = *shadow_points_ptr;
  for (int y=0; y< (int)range_image.height; ++y)
  {
    for (int x=0; x< (int)range_image.width; ++x)
    {
      if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER])
        border_points.points.push_back (range_image.points[y*range_image.width + x]);
      if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT])
        veil_points.points.push_back (range_image.points[y*range_image.width + x]);
      if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__SHADOW_BORDER])
        shadow_points.points.push_back (range_image.points[y*range_image.width + x]);
    }
  }

  std::cout << "border_points: " << border_points.size() << std::endl;
  std::cout << "veil_points: " << veil_points.size() << std::endl;
  std::cout << "shadow_points: " << shadow_points.size() << std::endl;
  cout << "Border Extraction - No Floor: " <<(double)(clock() - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return border_points_ptr;
  
}
}