#include "include/core/Map.h"

using namespace std;

namespace EllipsoidSLAM
{

Map::Map() {
    mCameraState = new g2o::SE3Quat();
}

void Map::addPoint(EllipsoidSLAM::PointXYZRGB *pPoint) {
    unique_lock<mutex> lock(mMutexMap);
    mspPoints.insert(pPoint);
}

void Map::addPointCloud(EllipsoidSLAM::PointCloud *pPointCloud) {
    unique_lock<mutex> lock(mMutexMap);

    for(auto iter=pPointCloud->begin();iter!=pPointCloud->end();++iter){
        mspPoints.insert(&(*iter));
    }
}

void Map::clearPointCloud() {
    unique_lock<mutex> lock(mMutexMap);

    mspPoints.clear();
}

void Map::addEllipsoid(ellipsoid *pObj)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoids.push_back(pObj);
}


vector<ellipsoid*> Map::GetAllEllipsoids()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspEllipsoids;
}

std::vector<PointXYZRGB*> Map::GetAllPoints() {
    unique_lock<mutex> lock(mMutexMap);
    return vector<PointXYZRGB*>(mspPoints.begin(),mspPoints.end());
}

void Map::addPlane(plane *pPlane)
{
    unique_lock<mutex> lock(mMutexMap);
    mspPlanes.insert(pPlane);
}


vector<plane*> Map::GetAllPlanes()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<plane*>(mspPlanes.begin(),mspPlanes.end());
}

void Map::setCameraState(g2o::SE3Quat* state) {
    unique_lock<mutex> lock(mMutexMap);
    mCameraState =  state;
}

void Map::addCameraStateToTrajectory(g2o::SE3Quat* state) {
    unique_lock<mutex> lock(mMutexMap);
    mvCameraStates.push_back(state);
}

g2o::SE3Quat* Map::getCameraState() {
    unique_lock<mutex> lock(mMutexMap);
    return mCameraState;
}

std::vector<g2o::SE3Quat*> Map::getCameraStateTrajectory() {
    unique_lock<mutex> lock(mMutexMap);
    return mvCameraStates;
}

std::vector<ellipsoid*> Map::getEllipsoidsUsingLabel(int label) {
    unique_lock<mutex> lock(mMutexMap);

    std::vector<ellipsoid*> mvpObjects;
    auto iter = mspEllipsoids.begin();
    for(; iter!=mspEllipsoids.end(); iter++)
    {

        if( (*iter)->miLabel == label )
            mvpObjects.push_back(*iter);

    }

    return mvpObjects;
}

std::map<int, ellipsoid*> Map::GetAllEllipsoidsMap() {
    std::map<int, ellipsoid*> maps;
    for(auto iter= mspEllipsoids.begin(); iter!=mspEllipsoids.end();iter++)
    {
        maps.insert(make_pair((*iter)->miInstanceID, *iter));
    }
    return maps;
}

void Map::clearPlanes(){
    unique_lock<mutex> lock(mMutexMap);
    mspPlanes.clear();
}

void Map::addEllipsoidVisual(ellipsoid *pObj)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsVisual.push_back(pObj);
}


vector<ellipsoid*> Map::GetAllEllipsoidsVisual()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspEllipsoidsVisual;
}

void Map::ClearEllipsoidsVisual()
{
    unique_lock<mutex> lock(mMutexMap);
    mspEllipsoidsVisual.clear();
}

bool Map::AddPointCloudList(const string& name, PointCloud* pCloud, int type){
    unique_lock<mutex> lock(mMutexMap);

    // Check repetition
    if(mmPointCloudLists.find(name) != mmPointCloudLists.end() )
    {
        // Exist
        
        if( type == 0){
            // replace it.
            mmPointCloudLists[name]->clear(); // release it
            mmPointCloudLists[name] = pCloud;
        }
        else if( type == 1 )
        {
            // add together
            for( auto &p : *pCloud )
                mmPointCloudLists[name]->push_back(p);
        }

        return false;
    }
    else{
        mmPointCloudLists.insert(make_pair(name, pCloud));
        return true;
    }
        
}

bool Map::DeletePointCloudList(const string& name, int type){
    unique_lock<mutex> lock(mMutexMap);

    if( type == 0 ) // complete matching: the name must be the same
    {
        auto iter = mmPointCloudLists.find(name);
        if (iter != mmPointCloudLists.end() )
        {
            mmPointCloudLists.erase(iter);
            return true;
        }
        else{
            std::cerr << "PointCloud name " << name << " doesn't exsit. Can't delete it." << std::endl;
            return false;
        }
    }
    else if ( type == 1 ) // partial matching
    {
        bool deleteSome = false;
        for( auto iter = mmPointCloudLists.begin();iter!=mmPointCloudLists.end();iter++ )
        {
            auto strPoints = *iter;
            if( strPoints.first.find(name) != strPoints.first.npos )
            {
                mmPointCloudLists.erase(iter);
                deleteSome = true;
            }
        }
        return deleteSome;
    }
    
}

bool Map::ClearPointCloudLists(){
    unique_lock<mutex> lock(mMutexMap);

    mmPointCloudLists.clear();
}

std::map<string, PointCloud*> Map::GetPointCloudList(){
    unique_lock<mutex> lock(mMutexMap);
    return mmPointCloudLists;
}

} // namespace 