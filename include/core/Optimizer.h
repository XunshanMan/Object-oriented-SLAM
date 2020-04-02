#ifndef ELLIPSOIDSLAM_OPTIMIZER_H
#define ELLIPSOIDSLAM_OPTIMIZER_H

#include "Frame.h"
#include "Map.h"
#include "Initializer.h"

#include "src/symmetry/Symmetry.h"

namespace EllipsoidSLAM {
    class Frame;
    class Map;
    class Optimizer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Optimizer();

        void GlobalObjectGraphOptimization(std::vector<Frame *> &pFrames, Map *pMap,
                                                 int rows, int cols, Matrix3d &mCalib,  std::map<int, Observations>& objectObservations,
                                                bool save_graph = false, bool withAssociation = false, bool check_visibility = false );

        void SetGroundPlane(Vector4d& normal);

    private:
        std::map<int, std::vector<float>> mMapObjectConstrain;

        bool mbGroundPlaneSet;
        Vector4d mGroundPlaneNormal;
    };

}

#endif //ELLIPSOIDSLAM_OPTIMIZER_H
