#include "include/core/Ellipsoid.h"

#include "src/Polygon/Polygon.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

namespace g2o
{
    ellipsoid::ellipsoid()
    {
    }

    // xyz roll pitch yaw half_scale
    void ellipsoid::fromMinimalVector(const Vector9d& v){
        Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3),v(4),v(5));
        pose = SE3Quat(posequat, v.head<3>());
        scale = v.tail<3>();

        vec_minimal = v;
    }

    // xyz quaternion, half_scale
    void ellipsoid::fromVector(const Vector10d& v){
        pose.fromVector(v.head<7>());
        scale = v.tail<3>();
        vec_minimal = toMinimalVector();
    }

    const Vector3d& ellipsoid::translation() const {return pose.translation();}
    void ellipsoid::setTranslation(const Vector3d& t_) {pose.setTranslation(t_);}
    void ellipsoid::setRotation(const Quaterniond& r_) {pose.setRotation(r_);}
    void ellipsoid::setRotation(const Matrix3d& R) {pose.setRotation(Quaterniond(R));}
    void ellipsoid::setScale(const Vector3d &scale_) {scale=scale_;}

    // apply update to current ellipsoid. exponential map
    ellipsoid ellipsoid::exp_update(const Vector9d& update)
    {
        ellipsoid res;
        res.pose = this->pose*SE3Quat::exp(update.head<6>());
        res.scale = this->scale + update.tail<3>();

        res.UpdateValueFrom(*this);
        res.vec_minimal = res.toMinimalVector();
        return res;
    }

    // TOBE DELETED.
    ellipsoid ellipsoid::exp_update_XYZABC(const Vector6d& update)
    {
        ellipsoid res;

        Vector6d pose_vec; pose_vec << 0, 0, 0, update[0], update[1], update[2];
        res.pose = this->pose*SE3Quat::exp(pose_vec); 
        res.scale = this->scale + update.tail<3>();

        res.UpdateValueFrom(*this);
        res.vec_minimal = res.toMinimalVector();
        return res;
    }

    Vector9d ellipsoid::ellipsoid_log_error_9dof(const ellipsoid& newone) const
    {
        Vector9d res;
        SE3Quat pose_diff = newone.pose.inverse()*this->pose;

        res.head<6>() = pose_diff.log(); 
        res.tail<3>() = this->scale - newone.scale; 
        return res;        
    }

    // change front face by rotate along current body z axis. 
    // another way of representing cuboid. representing same cuboid (IOU always 1)
    ellipsoid ellipsoid::rotate_ellipsoid(double yaw_angle) const // to deal with different front surface of cuboids
    {
        ellipsoid res;
        SE3Quat rot(Eigen::Quaterniond(cos(yaw_angle*0.5),0,0,sin(yaw_angle*0.5)),Vector3d(0,0,0));   // change yaw to rotation.
        res.pose = this->pose*rot;
        res.scale = this->scale;
        
        res.UpdateValueFrom(*this);
        res.vec_minimal = res.toMinimalVector();

        const double eps = 1e-6;
        if ( (std::abs(yaw_angle-M_PI/2.0) < eps) || (std::abs(yaw_angle+M_PI/2.0) < eps) || (std::abs(yaw_angle-3*M_PI/2.0) < eps))
            std::swap(res.scale(0),res.scale(1));   

        return res;
    }

    Vector9d ellipsoid::min_log_error_9dof(const ellipsoid& newone, bool print_details) const
    {
        bool whether_rotate_ellipsoid=true;  // whether rotate cube to find smallest error
        if (!whether_rotate_ellipsoid)
            return ellipsoid_log_error_9dof(newone);

        // NOTE rotating ellipsoid... since we cannot determine the front face consistenly, different front faces indicate different yaw, scale representation.
        // need to rotate all 360 degrees (global cube might be quite different from local cube)
        // this requires the sequential object insertion. In this case, object yaw practically should not change much. If we observe a jump, we can use code
        // here to adjust the yaw.
        Vector4d rotate_errors_norm; Vector4d rotate_angles(-1,0,1,2); // rotate -90 0 90 180
        Eigen::Matrix<double, 9, 4> rotate_errors;
        for (int i=0;i<rotate_errors_norm.rows();i++)
        {
            ellipsoid rotated_cuboid = newone.rotate_ellipsoid(rotate_angles(i)*M_PI/2.0);  // rotate new cuboids
            Vector9d cuboid_error = this->ellipsoid_log_error_9dof(rotated_cuboid);
            rotate_errors_norm(i) = cuboid_error.norm();
            rotate_errors.col(i) = cuboid_error;
        }
        int min_label;
        rotate_errors_norm.minCoeff(&min_label);
        if (print_details)
            if (min_label!=1)
                std::cout<<"Rotate ellipsoid   "<<min_label<<std::endl;
        return rotate_errors.col(min_label);
    }

    // transform a local cuboid to global cuboid  Twc is camera pose. from camera to world
    ellipsoid ellipsoid::transform_from(const SE3Quat& Twc) const{
        ellipsoid res;
        res.pose = Twc*this->pose;
        res.scale = this->scale;
        
        res.UpdateValueFrom(*this);
        res.vec_minimal = res.toMinimalVector();

        return res;
    }

    // transform a global cuboid to local cuboid  Twc is camera pose. from camera to world
    ellipsoid ellipsoid::transform_to(const SE3Quat& Twc) const{
        ellipsoid res;
        res.pose = Twc.inverse()*this->pose;
        res.scale = this->scale;

        res.UpdateValueFrom(*this);
        res.vec_minimal = res.toMinimalVector();
        
        return res;
    }

    // xyz roll pitch yaw half_scale
    Vector9d ellipsoid::toMinimalVector() const{
        Vector9d v;
        v.head<6>() = pose.toXYZPRYVector();
        v.tail<3>() = scale;
        return v;
    }

    // xyz quaternion, half_scale
    Vector10d ellipsoid::toVector() const{
        Vector10d v;
        v.head<7>() = pose.toVector();
        v.tail<3>() = scale;
        return v;
    }

    Matrix4d ellipsoid::similarityTransform() const
    {
        Matrix4d res = pose.to_homogeneous_matrix();    // 4x4 transform matrix
        Matrix3d scale_mat = scale.asDiagonal();
        res.topLeftCorner<3,3>() = res.topLeftCorner<3,3>()*scale_mat;
        return res;
    }


    void ellipsoid::UpdateValueFrom(const g2o::ellipsoid& e){
        this->miLabel = e.miLabel;
        this->mbColor = e.mbColor;
        this->mvColor = e.mvColor;
        this->miInstanceID = e.miInstanceID;

        this->prob = e.prob;
    }

    ellipsoid::ellipsoid(const g2o::ellipsoid &e) {
        pose = e.pose;
        scale = e.scale;
        vec_minimal = e.vec_minimal;

        UpdateValueFrom(e);
    }

    const ellipsoid& ellipsoid::operator=(const g2o::ellipsoid &e) {
        pose = e.pose;
        scale = e.scale;
        vec_minimal = e.vec_minimal;

        UpdateValueFrom(e);
        return e;
    }

    // ************* Functions As Ellipsoids ***************
    Vector2d ellipsoid::projectCenterIntoImagePoint(const SE3Quat& campose_cw, const Matrix3d& Kalib)
    {
        Matrix3Xd  P = generateProjectionMatrix(campose_cw, Kalib);

        Vector3d center_pos = pose.translation();
        Vector4d center_homo = real_to_homo_coord<double>(center_pos);
        Vector3d u_homo = P * center_homo;
        Vector2d u = homo_to_real_coord_vec<double>(u_homo);

        return u;
    }

    // project the ellipsoid into the image plane, and get an ellipse represented by a Vector5d.
    // Ellipse: x_c, y_c, theta, axis1, axis2
    Vector5d ellipsoid::projectOntoImageEllipse(const SE3Quat& campose_cw, const Matrix3d& Kalib) const 
    {
        Matrix4d Q_star = generateQuadric();
        Matrix3Xd  P = generateProjectionMatrix(campose_cw, Kalib);
        Matrix3d C_star = P * Q_star * P.transpose();
        Matrix3d C = C_star.inverse(); 
        C = C / C(2,2); // normalize

        SelfAdjointEigenSolver<Matrix3d> es(C);    // ascending sort by default
        VectorXd eigens = es.eigenvalues();

        // If it is an ellipse, the sign of eigen values must be :  1 1 -1 
        // Ref book : Multiple View Geometry in Computer Vision
        int num_pos = int(eigens(0)>0) +int(eigens(1)>0) +int(eigens(2)>0);
        int num_neg = int(eigens(0)<0) +int(eigens(1)<0) +int(eigens(2)<0);

        // matrix to equation coefficients: ax^2+bxy+cy^2+dx+ey+f=0
        double a = C(0,0);
        double b = C(0,1)*2;
        double c = C(1,1);
        double d = C(0,2)*2;
        double e = C(2,1)*2;
        double f = C(2,2);

        // get x_c, y_c, theta, axis1, axis2 from coefficients
        double delta = c*c - 4.0*a*b;
        double k = (a*f-e*e/4.0) - pow((2*a*e-c*d),2)/(4*(4*a*b-c*c));
        double theta = 1/2.0*atan2(b,(a-c));
        double x_c = (b*e-2*c*d)/(4*a*c-b*b);
        double y_c = (b*d-2*a*e)/(4*a*c-b*b);
        double a_2 =  2*(a* x_c*x_c+ c * y_c*y_c+ b *x_c*y_c -1) /(a + c + sqrt((a-c)*(a-c)+b*b));
        double b_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c -1) /( a + c - sqrt((a-c)*(a-c)+b*b));

        double axis1= sqrt(a_2);
        double axis2= sqrt(b_2);

        Vector5d output;
        output << x_c, y_c, theta, axis1, axis2;

        return output;
    }

    // Get the bounding box from ellipse in image plane
    Vector4d ellipsoid::getBoundingBoxFromEllipse(Vector5d &ellipse) const
    {
        double a = ellipse[3];
        double b = ellipse[4];
        double theta = ellipse[2];
        double x = ellipse[0];
        double y = ellipse[1];
        
        double cos_theta_2 = cos(theta)*cos(theta);
        double sin_theta_2 = 1- cos_theta_2;

        double x_limit = sqrt(a*a*cos_theta_2+b*b*sin_theta_2);
        double y_limit = sqrt(a*a*sin_theta_2+b*b*cos_theta_2);

        Vector4d output;
        output[0] = x-x_limit; // left up
        output[1] = y-y_limit;
        output[2] = x+x_limit; // right down
        output[3] = y+y_limit;

        return output;
    }

    // Get projection matrix P = K [ R | t ]
    Matrix3Xd ellipsoid::generateProjectionMatrix(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Matrix3Xd identity_lefttop;
        identity_lefttop.resize(3, 4);
        identity_lefttop.col(3)=Vector3d(0,0,0);
        identity_lefttop.topLeftCorner<3,3>() = Matrix3d::Identity(3,3);

        Matrix3Xd proj_mat = Kalib * identity_lefttop;
        proj_mat = proj_mat * campose_cw.to_homogeneous_matrix();

        return proj_mat;
    }

    // Get Q^*
    Matrix4d ellipsoid::generateQuadric() const
    {
        Vector4d axisVec;
        axisVec << 1/(scale[0]*scale[0]), 1/(scale[1]*scale[1]), 1/(scale[2]*scale[2]), -1;
        Matrix4d Q_c = axisVec.asDiagonal();  
        Matrix4d Q_c_star = Q_c.inverse();  
        Matrix4d Q_pose_matrix = pose.to_homogeneous_matrix();   // Twm  model in world,  world to model
        Matrix4d Q_c_star_trans = Q_pose_matrix * Q_c_star * Q_pose_matrix.transpose(); 

        return Q_c_star_trans;
    }

    // Get the projected bounding box in the image plane of the ellipsoid using a camera pose and a calibration matrix.
    Vector4d ellipsoid::getBoundingBoxFromProjection(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Vector5d ellipse = projectOntoImageEllipse(campose_cw, Kalib);
        return getBoundingBoxFromEllipse(ellipse);
    }

    Vector3d ellipsoid::getColor(){
        return mvColor.head(3);
    }

    Vector4d ellipsoid::getColorWithAlpha(){
        return mvColor;
    }

    void ellipsoid::setColor(const Vector3d &color_, double alpha){
        mbColor = true;
        mvColor.head<3>() = color_;
        mvColor[3] = alpha;

    }

    bool ellipsoid::isColorSet(){
        return mbColor;
    }

    bool ellipsoid::CheckObservability(const SE3Quat& campose_cw)
    {
        Vector3d ellipsoid_center = toMinimalVector().head(3);    // Pwo
        Vector4d center_homo = real_to_homo_coord_vec<double>(ellipsoid_center);

        Eigen::Matrix4d projMat = campose_cw.to_homogeneous_matrix(); // Tcw
        Vector4d center_inCameraAxis_homo = projMat * center_homo;   // Pco =  Tcw * Pwo
        Vector3d center_inCameraAxis = homo_to_real_coord_vec<double>(center_inCameraAxis_homo);

        if( center_inCameraAxis_homo(2) < 0)    // if the center is behind the camera ; z<0
        {
            return false;
        }
        else
            return true;
    }

    // calculate the IoU Error between two axis-aligned ellipsoid
    double ellipsoid::calculateMIoU(const g2o::ellipsoid& e) const
    {
        return calculateIntersectionError(*this, e);
    }

    double ellipsoid::calculateIntersectionOnZ(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const
    {
        g2o::SE3Quat pose_diff = e1.pose.inverse() * e2.pose;
        double z1 = 0; double z2 = pose_diff.translation()[2];

        bool flag_oneBigger = false;
        if( z1 > z2 )
            flag_oneBigger = true;

        double length;
        if( flag_oneBigger )
        {
            length = (z2 + e2.scale[2]) - (z1 - e1.scale[2]);   
        }
        else 
            length = (z1 + e1.scale[2]) - (z2 - e2.scale[2]);  

        if( length < 0 )
            length = 0;     // if they are not intersected
        
        return length;
    }

    double ellipsoid::calculateArea(const g2o::ellipsoid& e) const
    {
        return e.scale[0]*e.scale[1]*e.scale[2]*8;
    }

    void OutputPolygon(EllipsoidSLAM::Polygon& polygon, double resolution)
    {
        int num = polygon.n;
        for( int i=0;i<num;i++)
            std::cout << i << ":" << polygon[i].x << ", " << polygon[i].y << std::endl;
        std::cout << std::endl;
    }

    // Calculate the intersection area after projected the external cubes of two axis-aligned ellipsoids into XY-Plane.
    double ellipsoid::calculateIntersectionOnXY(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const
    {
        // First, get the axis-aligned pose error
        g2o::SE3Quat pose_diff = e1.pose.inverse() * e2.pose;
        double x_center1 = 0; double y_center1 = 0;

        double x_center2 = pose_diff.translation()[0];
        double y_center2 = pose_diff.translation()[1];

        double roll,pitch,yaw;
        quat_to_euler_zyx(pose_diff.rotation(),roll,pitch,yaw);

        double a1 = std::abs(e1.scale[0]);
        double b1 = std::abs(e1.scale[1]);

        double a2 = std::abs(e2.scale[0]);
        double b2 = std::abs(e2.scale[1]);

        // Use polygon to calculate the intersection
        EllipsoidSLAM::Polygon polygon1, polygon2;
        double resolution = 0.001;  // m / resolution = pixel
        polygon1.add(cv::Point(a1/resolution, b1/resolution));    // cvPoint only accepts integer, so use resolution to map meter to pixel ( 0.01 resolution means: 1pixel = 0.01m )
        polygon1.add(cv::Point(-a1/resolution, b1/resolution)); 
        polygon1.add(cv::Point(-a1/resolution, -b1/resolution)); 
        polygon1.add(cv::Point(a1/resolution, -b1/resolution)); 

        double c_length = sqrt(a2*a2+b2*b2);

        double init_theta = CV_PI/2.0 - atan2(a2,b2);
        Vector4d angle_plus_vec;
        angle_plus_vec << 0, atan2(a2,b2)*2, CV_PI, CV_PI+atan2(a2,b2)*2;
        for( int n=0;n<4;n++){
            double angle_plus = angle_plus_vec[n];  // rotate 90deg for four times
            double point_x = c_length * cos( init_theta - yaw + angle_plus ) + x_center2;
            double point_y = c_length * sin( init_theta - yaw + angle_plus ) + y_center2;
            polygon2.add(cv::Point(point_x/resolution, point_y/resolution));  
        }

        // calculate the intersection
        EllipsoidSLAM::Polygon interPolygon;
        EllipsoidSLAM::intersectPolygon(polygon1, polygon2, interPolygon);

        // eliminate resolution.
        double inter_area = interPolygon.area();
        double inter_area_in_m = inter_area * resolution * resolution;

        return inter_area_in_m;
    }

    double ellipsoid::calculateIntersectionError(const g2o::ellipsoid& e1, const g2o::ellipsoid& e2) const
    {
        //          AXB          
        // IoU = ----------
        //          AUB
        //   AXB  =  intersection
        //   AUB  =  A+B-intersection

        // Error of IoU : 1 - IoU
        double areaA = std::abs(calculateArea(e1));        
        std::cout << "areaA : " << areaA << std::endl;

        double areaB = std::abs(calculateArea(e2));
        std::cout << "areaB : " << areaB << std::endl;

        double proj_inter = calculateIntersectionOnXY(e1,e2);
        double z_inter = calculateIntersectionOnZ(e1,e2);
        std::cout << "projInter : " << proj_inter << std::endl;
        std::cout << "z_inter : " << z_inter << std::endl;

        double areaIntersection = proj_inter * z_inter;
        std::cout << "areaIntersection : " << areaIntersection << std::endl;

        double MIoU = 1 - ((areaIntersection) / (areaA + areaB - areaIntersection));
        std::cout << "MIoU : " << MIoU << std::endl;
        std::cout << "e1 : " << e1.toMinimalVector().transpose() << std::endl;
        std::cout << "e2 : " << e2.toMinimalVector().transpose() << std::endl;

        return MIoU;
    }

    // ***************** Functions as Cubes ******************

    // calculate the external cube of the ellipsoid
    // 8 corners 3*8 matrix, each row is x y z
    Matrix3Xd ellipsoid::compute3D_BoxCorner() const
    {
        Matrix3Xd corners_body;corners_body.resize(3,8);
        corners_body<< 1, 1, -1, -1, 1, 1, -1, -1,
                1, -1, -1, 1, 1, -1, -1, 1,
                -1, -1, -1, -1, 1, 1, 1, 1;
        Matrix3Xd corners_world = homo_to_real_coord<double>(similarityTransform()*real_to_homo_coord<double>(corners_body));
        return corners_world;
    }

    Matrix2Xd ellipsoid::projectOntoImageBoxCorner(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Matrix3Xd corners_3d_world = compute3D_BoxCorner();
        Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib*homo_to_real_coord<double>(campose_cw.to_homogeneous_matrix()*real_to_homo_coord<double>(corners_3d_world)));

        return corner_2d;
    }

    // get rectangles after projection  [topleft, bottomright]
    Vector4d ellipsoid::projectOntoImageRect(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Matrix2Xd corner_2d = projectOntoImageBoxCorner(campose_cw, Kalib);
        Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // x y
        Vector2d topleft = corner_2d.rowwise().minCoeff();
        return Vector4d(topleft(0),topleft(1),bottomright(0),bottomright(1));
    }

    // get rectangles after projection  [center, width, height]
    Vector4d ellipsoid::projectOntoImageBbox(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Vector4d rect_project = projectOntoImageRect(campose_cw, Kalib);  // top_left, bottom_right  x1 y1 x2 y2
        Vector2d rect_center = (rect_project.tail<2>()+rect_project.head<2>())/2;
        Vector2d widthheight = rect_project.tail<2>()-rect_project.head<2>();
        return Vector4d(rect_center(0),rect_center(1),widthheight(0),widthheight(1));
    }

} // g2o