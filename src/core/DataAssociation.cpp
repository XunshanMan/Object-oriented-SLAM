#include "core/DataAssociation.h"

#include <iostream>
#include <map>

namespace EllipsoidSLAM
{

    DataAssociationSolver::DataAssociationSolver(Map* pMap)
    {
        mpMap = pMap;

        miInstanceNum = 0;
    }

    std::vector<int> DataAssociationSolver::Solve(EllipsoidSLAM::Frame* pFrame, bool mb3D)
    {
        Eigen::MatrixXd &obsMat = pFrame->mmObservations;
        g2o::SE3Quat campose_wc = pFrame->cam_pose_Twc;

        auto instanceObjMap = mpMap->GetAllEllipsoidsMap();
        int num_object = instanceObjMap.size();

        int num_obs = obsMat.rows();

        std::map<int, g2o::ellipsoid*> valid3DObsMap;

        std::vector<int> associations; associations.resize(num_obs);
        if( mb3D )  // only support 3d observations
        {
            // filter invalid 3d observaions
            for( int i=0;i<num_obs;i++)
            {
                g2o::ellipsoid* pObjObserved = pFrame->mpLocalObjects[i];
                if(pObjObserved != NULL ) 
                {
                    valid3DObsMap.insert(make_pair(i, pObjObserved));
                }
                else
                    associations[i] = -1;
            }

            // construct a new list for those valid observations
            int validObNum = valid3DObsMap.size();

            if( validObNum < 1 ) return associations;

            // calculate the costMat
            MatrixXd costMat; costMat.resize(validObNum, num_object);     // row : observations , col : objects in the map

            int objObservedId = 0;
            for( auto& validIdObPair : valid3DObsMap )
            {
                g2o::ellipsoid* pObjObserved = validIdObPair.second;
                g2o::ellipsoid pObjObservedWorld = pObjObserved->transform_from(campose_wc);
                VectorXd costVec; costVec.resize(num_object);

                int objInMapId = 0;
                for( auto& instanceObj : instanceObjMap )
                {
                    g2o::ellipsoid* pObj = instanceObj.second;
                    // calculate the distance of between their centers
                    double distance = (pObj->pose.translation() - pObjObservedWorld.pose.translation()).norm();

                    double cost = distance;   
                    costVec[objInMapId] = cost;
                    objInMapId ++;
                }

                costMat.row(objObservedId) = costVec.transpose();
                objObservedId++;
            }

            // solve results from costMat
            std::vector<int> matAssociations = SolveCostMat(costMat);

            // save the association results of the valid observations to output
            objObservedId = 0;
            for( auto& validIdObPair : valid3DObsMap )
            {
                associations[validIdObPair.first] = matAssociations[objObservedId];
                objObservedId++;
            }
            return associations;
        }
    }

    // Solve associations from every rows to every cols separately.
    // If an observation doesn't belong to any object in the map, it will get a new instance ID.
    std::vector<int> DataAssociationSolver::SolveCostMat(MatrixXd& mat){
        // Method : Iterate through each row in turn, taking the column with minimum cost value if the value is also less than the thresh,
        // and delete the column from the cost matrix. Repeat the process for every row until there are no columns left.
        // ----- Settings ---
        double SETTING_DIS_THRESH = 1.0;    // the minimum cost value must be less than the thresh to be considered valid.

        MatrixXd costMat = mat;
        int rows = costMat.rows();

        std::vector<int> matAssocitions; matAssocitions.resize(rows);

        for( int i=0; i<rows; i++ )
        {
            Eigen::VectorXd vec = costMat.row(i);

            if( vec.rows() < 1 ){
                // if there are no objects in the map, just initialize a new instance ID
                matAssocitions[i] = CreateInstance(); 
                cout << "For the first data association, create a new instance." << endl;
                continue;
            }

            // get the minimum cost value and its ID.
            MatrixXd::Index minRow, minCol;
            double min = vec.minCoeff(&minRow, &minCol);
            if( min < SETTING_DIS_THRESH )
            {
                int index = minRow; // row of the vec is corresponding to the column of the cost matrix
                // successful association
                matAssocitions[i] = index;

                // set the cost value that has been just associated to a big value
                VectorXd maxVec; maxVec.resize(rows);
                maxVec.setConstant(999);

                costMat.col(index) = maxVec;
            }
            else
            {
                // association fails, create a new instance
                matAssocitions[i] = CreateInstance();
            }
        }

        return matAssocitions;
    }

    int DataAssociationSolver::CreateInstance()
    {
        return miInstanceNum++;
    }

}