#include "include/core/Optimizer.h"
#include "include/core/Ellipsoid.h"
#include "include/core/BasicEllipsoidEdges.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <src/config/Config.h>

using namespace cv;

namespace EllipsoidSLAM
{

bool isInImage(Eigen::Vector2d& uv, int rows, int cols){
    
    if( uv(0) >0 && uv(0) < cols )
        if( uv(1) >0 && uv(1) < rows )
            return true;

    return false;
}

// [ Unused ]
// The function checks several conditions to decide whether the edge will be activated:
// 1) if the object is behind the camera
// 2) if the camera is inside the elipsoid, which generates an ill condition
// 3) if the projected ellipse lies in the image width and height
//    if one of the boundingbox vertces lies in the image, active it too.
bool checkVisibility(g2o::EdgeSE3EllipsoidProj *edge, g2o::VertexSE3Expmap *vSE3, 
    g2o::VertexEllipsoid *vEllipsoid, Eigen::Matrix3d &mCalib, int rows, int cols)
{
    g2o::ellipsoid e = vEllipsoid->estimate();
    Vector3d ellipsoid_center = e.toMinimalVector().head(3);    // Pwo
    Vector4d center_homo = real_to_homo_coord_vec<double>(ellipsoid_center);

    g2o::SE3Quat campose_Tcw = vSE3->estimate();
    Eigen::Matrix4d projMat = campose_Tcw.to_homogeneous_matrix(); // Tcw
    
    // project to image plane.
    Vector4d center_inCameraAxis_homo = projMat * center_homo;   // Pco =  Tcw * Pwo
    Vector3d center_inCameraAxis = homo_to_real_coord_vec<double>(center_inCameraAxis_homo);

    if( center_inCameraAxis_homo(2) < 0)    // if the object is behind the camera. z< 0
    {
        return false;
    }

    // check if the camera is inside the elipsoid, which generates an ill condition
    g2o::SE3Quat campose_Twc = campose_Tcw.inverse();
    Eigen::Vector3d X_cam = campose_Twc.translation();
    Eigen::Vector4d X_homo = real_to_homo_coord_vec<double>(X_cam);
    Eigen::Matrix4d Q_star = e.generateQuadric();
    Eigen::Matrix4d Q = Q_star.inverse();
    double point_in_Q = X_homo.transpose() * Q * X_homo;
    if(point_in_Q < 0)  // the center of the camera is inside the ellipsoid
        return false;

    // check if the projected ellipse lies in the image width and height
    Eigen::Matrix3Xd P = e.generateProjectionMatrix(campose_Tcw, mCalib);
    Eigen::Vector3d uv = P * center_homo;
    uv = uv/uv(2);
    Eigen::Vector2d uv_2d(uv(0), uv(1));
    if( isInImage(uv_2d, rows, cols) )
            return true;

    // if one of the boundingbox vertces lies in the image, active it too.
    Vector4d vec_proj = edge->getProject();
    Vector2d point_lu(vec_proj(0), vec_proj(1));
    Vector2d point_rd(vec_proj(2), vec_proj(3));
    if( isInImage(point_lu,rows,cols) || isInImage(point_rd, rows, cols) )
        return true;

    return false;

}

void Optimizer::GlobalObjectGraphOptimization(std::vector<Frame *> &pFrames, Map *pMap,
        int rows, int cols, Matrix3d &mCalib, std::map<int, Observations>& objectObservations, bool save_graph, bool withAssociation, bool check_visibility) {
    // ************************ LOAD CONFIGURATION ************************
    double config_ellipsoid_3d_scale = Config::Get<double>("Optimizer.Edges.3DEllipsoid.Scale");
    bool mbSetGravityPrior = Config::Get<int>("Optimizer.Edges.GravityPrior.Open") == 1;  
    double dGravityPriorScale = Config::Get<double>("Optimizer.Edges.GravityPrior.Scale");
    bool mbOpen3DProb = true;  

    // OUTPUT
    std::cout << " -- Optimization parameters : " << std::endl;
    if(mbGroundPlaneSet)
        std::cout << " [ Using Ground Plane: " << mGroundPlaneNormal.transpose() << " ] " << std::endl;

    if(!mbGroundPlaneSet || !mbSetGravityPrior )   
        std::cout << " * Gravity Prior : closed." << std::endl;
    else
        std::cout << " * Gravity Prior : Open." << std::endl;
    
    cout<<" * Scale_3dedge: " << config_ellipsoid_3d_scale << endl;
    cout<<" * Scale_GravityPrior: " << dGravityPriorScale << endl;
    // ************************************************************************

    // Initialize variables.
    int total_frame_number = int(pFrames.size());
    std::map<int, ellipsoid*> pEllipsoidsMapWithInstance = pMap->GetAllEllipsoidsMap();
    int objects_num = int(pEllipsoidsMapWithInstance.size());

    // initialize graph optimization.
    g2o::SparseOptimizer graph;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    graph.setAlgorithm(solver);
    graph.setVerbose(false);        // Set output.

    std::map<int, g2o::VertexEllipsoid*> vEllipsoidVertexMaps;
    std::vector<g2o::EdgeSE3EllipsoidProj*> edges, edgesValid, edgesInValid;
    std::vector<bool> validVec; validVec.resize(total_frame_number);
    std::vector<g2o::EdgeEllipsoidGravityPlanePrior *> edgesEllipsoidGravityPlanePrior;     // Gravity prior
    std::vector<g2o::VertexSE3Expmap*> vSE3Vertex;

    // Add SE3 vertices for camera poses
    bool bSLAM_mode = false;   // Mapping Mode : Fix camera poses and mapping ellipsoids only
    for( int frame_index=0; frame_index< total_frame_number ; frame_index++) {
        g2o::SE3Quat curr_cam_pose_Twc = pFrames[frame_index]->cam_pose_Twc;
        Eigen::MatrixXd det_mat = pFrames[frame_index]->mmObservations;

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setId(graph.vertices().size());
        graph.addVertex(vSE3);
        vSE3->setEstimate(pFrames[frame_index]->cam_pose_Tcw); // Tcw
        if(!bSLAM_mode)
            vSE3->setFixed(true);       // Fix all the poses in mapping mode.
        else 
            vSE3->setFixed(frame_index == 0);   
        vSE3Vertex.push_back(vSE3);

        // Add odom edges if in SLAM Mode
        if(bSLAM_mode && frame_index > 0){
            g2o::SE3Quat prev_cam_pose_Tcw = pFrames[frame_index-1]->cam_pose_Twc.inverse();
            g2o::SE3Quat curr_cam_pose_Tcw = curr_cam_pose_Twc.inverse();
            g2o::SE3Quat odom_val = curr_cam_pose_Tcw*prev_cam_pose_Tcw.inverse();;

            g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vSE3Vertex[frame_index-1] ));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vSE3Vertex[frame_index] ));
            e->setMeasurement(odom_val);

            e->setId(graph.edges().size());
            Vector6d inv_sigma;inv_sigma<<1,1,1,1,1,1;
            inv_sigma = inv_sigma*1.0;
            Matrix6d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
            e->setInformation(info);
            graph.addEdge(e);
        }
    }

    // Initialize objects vertices and add edges of camera-objects 2d observations 
    int objects_ob_num = objectObservations.size();
    int objectid_in_edge = 0;
    int current_ob_id = 0;  
    int symplaneid_in_edge = 0;
    for(auto instanceObs : objectObservations )        
    {
        int instance = instanceObs.first;
        if (pEllipsoidsMapWithInstance.find(instance) == pEllipsoidsMapWithInstance.end())  // if the instance has not been initialized yet
            continue;

        Observations obs = instanceObs.second;

        // Add objects vertices
        g2o::VertexEllipsoid *vEllipsoid = new g2o::VertexEllipsoid();
        vEllipsoid->setEstimate(*pEllipsoidsMapWithInstance[instance]);
        vEllipsoid->setId(graph.vertices().size());
        vEllipsoid->setFixed(false);
        graph.addVertex(vEllipsoid);
        vEllipsoidVertexMaps.insert(make_pair(instance, vEllipsoid));

        // Add gravity prior
        if(mbGroundPlaneSet && mbSetGravityPrior ){
            g2o::EdgeEllipsoidGravityPlanePrior *vGravityPriorEdge = new g2o::EdgeEllipsoidGravityPlanePrior;
            vGravityPriorEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>( vEllipsoid ));

            vGravityPriorEdge->setMeasurement(mGroundPlaneNormal);  
            Matrix<double,1,1> inv_sigma;
            inv_sigma << 1 * dGravityPriorScale;
            MatrixXd info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
            vGravityPriorEdge->setInformation(info);
            
            graph.addEdge(vGravityPriorEdge);
            edgesEllipsoidGravityPlanePrior.push_back(vGravityPriorEdge);
            
        }

        // Add camera-objects 2d constraints 
        // At least 3 observations are needed to activiate 2d edges. Since one ellipsoid has 9 degrees,
        // and every observation offers 4 constrains, only at least 3 observations could fully constrain an ellipsoid.
        bool bVvalid_2d_constraints = (obs.size() > 2);
        if( bVvalid_2d_constraints ){
            for( int i=0; i<obs.size(); i++ )
            {
                Observation* vOb = obs[i];
                int label = vOb->label;
                Vector4d measure_bbox = vOb->bbox;
                double measure_prob = vOb->rate;

                // find the corresponding frame vertex.
                int frame_index = vOb->pFrame->frame_seq_id;
                g2o::VertexSE3Expmap *vSE3 = vSE3Vertex[frame_index]; 
                // create 2d edge
                g2o::EdgeSE3EllipsoidProj *e = new g2o::EdgeSE3EllipsoidProj();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>( vSE3 ));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>( vEllipsoidVertexMaps[instance] ));
                e->setMeasurement(measure_bbox); 
                e->setId(graph.edges().size());    
                Vector4d inv_sigma;
                inv_sigma << 1, 1, 1, 1;   
                Matrix4d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
                info = info * measure_prob;
                e->setInformation(info);
                // e->setRobustKernel(new g2o::RobustKernelHuber());     // Huber Kernel
                e->setKalib(mCalib);
                e->setLevel(0);
                edges.push_back(e);

                // Two conditions for valid 2d edges:
                // 1) [closed] visibility check: see the comments of function checkVisibility for detail.
                // 2) NaN check
                bool c1 = (!check_visibility) || checkVisibility(e, vSE3, vEllipsoidVertexMaps[instance], mCalib, rows, cols);

                e->computeError();
                double e_error = e->chi2();
                bool c2 = !isnan(e_error);  // NaN check

                if( c1 && c2 ){
                    graph.addEdge(e);
                    edgesValid.push_back(e);  // valid edges and invalid edges are for debug output
                }
                else
                    edgesInValid.push_back(e);   
            }
        }
    }

    // Add 3d constraints
    std::vector<g2o::EdgeSE3Ellipsoid9DOF*> vEllipsoid3DEdges;
    for( int frame_index=0; frame_index< total_frame_number ; frame_index++) {
        auto &ObjectsInFrame = pFrames[frame_index]->mpLocalObjects;
        for( auto obj : ObjectsInFrame )
        {
            if( obj == NULL ) continue;
            int instance = obj->miInstanceID;
            if(vEllipsoidVertexMaps.find(instance) == vEllipsoidVertexMaps.end())
                continue;   // if the instance has not been initialized
            
            auto vEllipsoid = vEllipsoidVertexMaps[instance];  
            auto vSE3 = vSE3Vertex[frame_index];

            // create 3d edges
            g2o::EdgeSE3Ellipsoid9DOF* vEllipsoid3D = new g2o::EdgeSE3Ellipsoid9DOF; 
            vEllipsoid3D->setId(graph.edges().size()); 
            vEllipsoid3D->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>( vSE3 ));
            vEllipsoid3D->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>( vEllipsoidVertexMaps[instance] ));                
            vEllipsoid3D->setMeasurement(*obj); 

            Vector9d inv_sigma;
            inv_sigma << 1,1,1,1,1,1,1,1,1;
            if(mbOpen3DProb)
                inv_sigma = inv_sigma * sqrt(obj->prob);
            Matrix9d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal() * config_ellipsoid_3d_scale;
            vEllipsoid3D->setInformation(info);

            graph.addEdge(vEllipsoid3D);
            vEllipsoid3DEdges.push_back(vEllipsoid3D);
        }
    }

    // output 
    std::cout << " -- GRAPH INFORMATION : " << std::endl;
    cout << " * Object Num : " << objects_num << endl;
    cout<<" * Vertices: "<<graph.vertices().size()<<endl;
    cout<<" * 2d Edges [Valid/Invalid] : " << edges.size() << " [" << edgesValid.size() <<"/" << edgesInValid.size() << "]" << endl;
    std::cout << " * 3d Edges : " << vEllipsoid3DEdges.size() << std::endl;
    cout<<" * Gravity edges: " << edgesEllipsoidGravityPlanePrior.size() << endl;
    cout << endl;

    graph.initializeOptimization();
    graph.optimize( 10 );  //optimization step

    // Update estimated ellispoids to the map
    for(auto iter=pEllipsoidsMapWithInstance.begin(); iter!=pEllipsoidsMapWithInstance.end();iter++)
    {
        if(vEllipsoidVertexMaps.find((*iter).first) != vEllipsoidVertexMaps.end())
        {
            *((*iter).second) = vEllipsoidVertexMaps[(*iter).first]->estimate();
        }
        else
        {
            cerr << "Find Error in ellipsoid storage. " << endl;
            cout << "Map: instance" << (*iter).first << endl;
        }
    }

    // Output optimization information.
    // object list
    ofstream out_obj("./object_list.txt");
    auto iter = vEllipsoidVertexMaps.begin();
    for(;iter!=vEllipsoidVertexMaps.end();iter++)
    {
        out_obj << iter->first << "\t" << iter->second->estimate().toMinimalVector().transpose() 
            << "\t" << iter->second->estimate().miLabel << std::endl;
    }
    out_obj.close();
}

Optimizer::Optimizer()
{
    mbGroundPlaneSet = false;
}

void Optimizer::SetGroundPlane(Vector4d& normal){
    mbGroundPlaneSet = true;
    mGroundPlaneNormal = normal;
}

}
