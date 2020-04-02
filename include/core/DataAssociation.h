#ifndef ELLIPSOIDSLAM_DATAASSOCIATION_H
#define ELLIPSOIDSLAM_DATAASSOCIATION_H

#include <core/Map.h>
#include <core/Frame.h>
#include <core/Ellipsoid.h>

namespace EllipsoidSLAM
{
    
class DataAssociationSolver
{

public:
    DataAssociationSolver(Map* pMap);   // the solver need the ellipsoids in the map

    std::vector<int> Solve(EllipsoidSLAM::Frame* pFrame, bool mb3D);    
private:
    std::vector<int> SolveCostMat(MatrixXd& mat);

    int CreateInstance();   // return the new instance ID

    Map* mpMap;

    bool mbWithAssociation;

    int miInstanceNum;
};

}

#endif // ELLIPSOIDSLAM_DATAASSOCIATION_H