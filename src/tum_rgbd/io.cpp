// Basic input and output of the TUM-RGBD dataset.

#include "io.h"
#include "utils/dataprocess_utils.h"

#include <iostream>
#include <string>


using namespace std;
namespace TUMRGBD
{

    void Dataset::loadDataset(string &path){
        cout << "Load dataset from: " << path << endl;
        
        msDatasetDir = path;
        msRGBDir = msDatasetDir + "rgb/";
        msDepthDir = msDatasetDir + "depth/";
        msGroundtruthPath = msDatasetDir + "groundtruth.txt";
        msAssociatePath = msDatasetDir + "associate.txt";
        msAssociateGroundtruthPath = msDatasetDir + "associateGroundtruth.txt";

        // get all the file names under the directory
        GetFileNamesUnderDir(msRGBDir, mvRGBFileNames);
        sortFileNames(mvRGBFileNames, mvRGBFileNames);
        miTotalNum = mvRGBFileNames.size();

        // generate the map between the timestamps to the depth images
        loadGroundTruthToMap(msGroundtruthPath);

        // generate the index from ID to the timestamps of rgb images
        generateIndexIdToRGBTimeStamp();        
        // generate the associations from rgb Timestamp to depth Timestamp
        LoadAssociationRGBToDepth(msAssociatePath);
        // generate the associations from rgb timestamp to the groundtruth
        LoadAssociationRGBToGroundtruth(msAssociateGroundtruthPath);

        // get debug file num:
        std::cout << "mmTimeStampToPose: " << mmTimeStampToPose.size() << std::endl;
        std::cout << "mvIdToDepthImagePath: " << mvIdToDepthImagePath.size() << std::endl;
        std::cout << "mvIdToGroundtruthTimeStamp: " << mvIdToGroundtruthTimeStamp.size() << std::endl;
        

        miCurrentID = 0;
        mbDetectionLoaded = false;
        mbOdomSet = false; 
    }

    bool Dataset::readFrame(cv::Mat &rgb, cv::Mat &depth, Eigen::VectorXd &pose){
        if(miCurrentID < miTotalNum) {
            miCurrentID++;
            bool result = findFrameUsingID(miCurrentID-1, rgb, depth, pose);
            return result;
        }
        else
        {
            std::cout << "[Dataset] no data left." << std::endl;
            return false;
        }
    }

    bool Dataset::findFrameUsingID(int id, cv::Mat &rgb, cv::Mat &depth, Eigen::VectorXd &pose){
        if( id <0 || id>=miTotalNum) return false;
        
        int currentID = id;
        string depthTimeStampAssociated = mvIdToDepthTimeStamp[currentID];
        if(depthTimeStampAssociated == ""){
            std::cout << "[Dataset] fail to load the depth timestamp." << std::endl;
            return false;  
        }

        string depthPath = msDatasetDir + "/" + mvIdToDepthImagePath[currentID];

        string gtTimeStamp = mvIdToGroundtruthTimeStamp[currentID];

        bool bFindPose = false;
        
        if( !mbOdomSet )    // if the odometry is set, return the pose of the odometry instead of the groundtruth
            bFindPose = getPoseFromTimeStamp(gtTimeStamp, pose);
        else
            bFindPose = getPoseFromRGBTimeStamp(mvIdToRGBTimeStamp[currentID], pose); 

        if(bFindPose){
            rgb = cv::imread(mvRGBFileNames[currentID], IMREAD_UNCHANGED);
            depth = cv::imread(depthPath, IMREAD_UNCHANGED);
            return true;
        }
        else
        {
            std::cout << "[Dataset] fail to find the pose ." << std::endl;
            return false;   
        }
    }


    map<string, string>::const_iterator AssociateWithNumber(const map<string, string>& map, const string &timestamp){
        auto iter = map.begin();
        for( ; iter != map.end(); iter++ )
        {
            if( iter->first == "" || timestamp == "" ) continue;
            if( std::abs(atof(iter->first.c_str()) - atof(timestamp.c_str())) < 0.001 )
            {
                // get it
                return iter;
            }
        }
        return iter;
    }

    map<string, VectorXd>::const_iterator AssociateWithNumber(const map<string, VectorXd>& map, const string &timestamp){
        auto iter = map.begin();
        for( ; iter != map.end(); iter++ )
        {
            if( iter->first == "" || timestamp == "" ) continue;
            if( std::abs(atof(iter->first.c_str()) - atof(timestamp.c_str())) < 0.001 )
            {
                // get it
                return iter;
            }
        }
        return iter;
    }

    // get pose from the map using the timestamp as index.
    // attention: this version uses string type of timestamp to search, which means their timestamp MUST be totally the same.
    //      it will be updated to double type in the future.
    bool Dataset::getPoseFromTimeStamp(string &timestamp, VectorXd &pose){
        for( auto iter : mmTimeStampToPose )
        {
            if( std::abs(std::atof(iter.first.c_str()) - std::atof(timestamp.c_str())) < 0.001 )
            {
                pose = iter.second;
                return true;
            }
        }

        return false;        
    }

    bool Dataset::getPoseFromRGBTimeStamp(string &timestamp, VectorXd &pose){
        auto iter = AssociateWithNumber(mmOdomRGBStampToPose, timestamp);
        // auto iter = mmOdomRGBStampToPose.find(timestamp);
        if( iter != mmOdomRGBStampToPose.end() )
        {
            pose = iter->second;
            return true;
        }
        else
        {
            return false;
        }
        
    }

    void Dataset::loadGroundTruthToMap(string &path){
        std::vector<std::vector<std::string>> strMat = readStringFromFile(path.c_str(), 0);

        int totalPose = strMat.size();
        for(int i=0;i<totalPose; i++)
        {
            std::vector<std::string> strVec = strMat[i];
            string timestamp = strVec[0];

            VectorXd pose; pose.resize(7);
            for(int p=1;p<8;p++)
                pose(p-1) = stod(strVec[p]);
            
            mmTimeStampToPose.insert(make_pair(timestamp, pose));
        }

    }

    void Dataset::LoadAssociationRGBToDepth(string &path)
    {
        std::vector<std::vector<std::string>> associateMat = readStringFromFile(msAssociatePath.c_str());

        map<string, string> mapRGBToDepth;
        map<string, string> mapRGBToDepthImagePath;
        int associationNum = associateMat.size();
        for(int i=0;i<associationNum; i++)
        {
            std::vector<std::string> lineVec = associateMat[i];
            string rgbTS = lineVec[0];
            string depthTS = lineVec[2];
            mapRGBToDepth.insert(make_pair(rgbTS, depthTS));
            mapRGBToDepthImagePath.insert(make_pair(rgbTS, lineVec[3]));
        }

        // for every rgb timestamp, find an associated depth timestamp
        mvIdToDepthTimeStamp.resize(miTotalNum);
        mvIdToDepthImagePath.resize(miTotalNum);
        for( int p=0;p<miTotalNum;p++)
        {
            auto iter = AssociateWithNumber(mapRGBToDepth, mvIdToRGBTimeStamp[p]);
            // auto iter = mapRGBToDepth.find(mvIdToRGBTimeStamp[p]);
            if(iter!=mapRGBToDepth.end())
            {
                mvIdToDepthTimeStamp[p] = iter->second;
            }
            else
            {
                mvIdToDepthTimeStamp[p] = "";   // empty stands for null
            }
            mvIdToDepthImagePath[p] = mapRGBToDepthImagePath[iter->first];
        }
    }

    void Dataset::LoadAssociationRGBToGroundtruth(string &path)
    {
        std::vector<std::vector<std::string>> associateMat = readStringFromFile(msAssociateGroundtruthPath.c_str());

        map<string, string> mapRGBToGt;
        int associationNum = associateMat.size();
        for(int i=0;i<associationNum; i++)
        {
            std::vector<std::string> lineVec = associateMat[i];
            string rgbTS = lineVec[0];
            string gtTS = lineVec[2];

            // Considering the precision of the timestamps of the result from the associate.py in TUM-RGB-D dataset,
            // we need to eliminate two zeros in the tails to make the groundtruth and the association have the same precision
            gtTS = gtTS.substr(0, gtTS.length()-2);

            mapRGBToGt.insert(make_pair(rgbTS, gtTS));
        }

        // for every rgb timestamp, find an associated groundtruth timestamp
        mvIdToGroundtruthTimeStamp.resize(miTotalNum);
        for( int p=0;p<miTotalNum;p++)
        {
            auto iter = AssociateWithNumber(mapRGBToGt, mvIdToRGBTimeStamp[p]);
            // auto iter = mapRGBToGt.find(mvIdToRGBTimeStamp[p]);
            if(iter!=mapRGBToGt.end())
            {
                mvIdToGroundtruthTimeStamp[p] = iter->second;
            }
            else
            {
                mvIdToGroundtruthTimeStamp[p] = "";   // empty stands for null
            }
            
        }
    }

    void Dataset::generateIndexIdToRGBTimeStamp(){
        // extract the bare name from a full path
        mvIdToRGBTimeStamp.clear();
        for(auto s:mvRGBFileNames)
        {
            string bareName = splitFileNameFromFullDir(s, true);
            mvIdToRGBTimeStamp.push_back(bareName);
        }
    }

    bool Dataset::empty(){
        return miCurrentID >= miTotalNum;
    }

    int Dataset::getCurrentID(){
        return miCurrentID;
    }

    int Dataset::getTotalNum()
    {
        return miTotalNum;
    }

    bool Dataset::loadDetectionDir(string &path)
    {
        msDetectionDir = path;
        mbDetectionLoaded = true;
    }

    Eigen::MatrixXd Dataset::getDetectionMat(){
        Eigen::MatrixXd detMat;
        if(!mbDetectionLoaded){
            std::cerr << "Detection dir has not loaded yet." << std::endl;
            return detMat;
        }
        // get the RGB timestamp as the name of the object detection file
        string bairName = mvIdToRGBTimeStamp[miCurrentID-1];
        string fullPath = msDetectionDir + bairName + ".txt";

        detMat = readDataFromFile(fullPath.c_str());
        return detMat;
        
    }

    vector<int> Dataset::generateValidVector(){
        vector<int> validVec;
        for(int i=0; i<miTotalNum; i++)
        {
            if(judgeValid(i)) 
                validVec.push_back(i);
        }

        return validVec;
    }

    bool Dataset::judgeValid(int id)
    {
        if( id <0 || id>=miTotalNum) return false;
        
        int currentID = id;
        
        string depthTimeStampAssociated = mvIdToDepthTimeStamp[currentID];
        if(depthTimeStampAssociated == ""){
            std::cout << "No depthTimeStampAssociated. " << std::endl;
            return false;   
        }

        string gtTimeStamp = mvIdToGroundtruthTimeStamp[currentID];

        VectorXd pose;
        if(getPoseFromTimeStamp(gtTimeStamp, pose))
            return true;
        else
        {
            return false;  
        }
            
    }

    bool Dataset::SetOdometry(const string& dir_odom, bool calibrate){

        std::vector<std::vector<std::string>> strMat = readStringFromFile(dir_odom.c_str(), 0);

        if(strMat.size() == 0) {
            std::cerr << " Odometry dir error! Keep gt. " << std::endl;
            return false;
        }

        mmOdomRGBStampToPose.clear();
        int totalPose = strMat.size();

        for(int i=0;i<totalPose; i++)
        {
            std::vector<std::string> strVec = strMat[i];
            string timestamp = strVec[0];

            VectorXd pose; pose.resize(7);
            for(int p=1;p<8;p++)
                pose(p-1) = stod(strVec[p]);
            
            mmOdomRGBStampToPose.insert(make_pair(timestamp, pose));
        }
        std::cout << "Setting odometry succeeds.";


        if( calibrate )
        {
            std::cout << "Get calibrate transform... " << std::endl;

            // find the corresponding groundtruth of the first timestamp of the odometry
            bool findCalibTrans = false;
            int transId = 0;
            for(auto timestampOdomPair: mmOdomRGBStampToPose)
            {
                string timestamp_gt = mvIdToGroundtruthTimeStamp[transId];  // assume that all the rgb images have corresponding odometry values
                
                assert( mvIdToRGBTimeStamp[transId] == timestampOdomPair.first && "Odom should start from the first rgb frame." );

                VectorXd gtPose;
                if( getPoseFromTimeStamp(timestamp_gt, gtPose) )
                {
                    g2o::SE3Quat pose_wc; pose_wc.fromVector(gtPose);
                    g2o::SE3Quat pose_oc; pose_oc.fromVector(timestampOdomPair.second);
                    mTransGtCalibrate = new g2o::SE3Quat;
                    *mTransGtCalibrate = pose_wc * pose_oc.inverse();

                    findCalibTrans = true;

                    break;
                }
                transId ++ ;
            }

            if( !findCalibTrans)
            {
                std::cerr << "Can't find calibrate transformation... Close calibraton!"<< std::endl;
                calibrate = false;
            }
            else
            {
                std::cout << "Find calibration trans ID: " << transId << std::endl;

                
                for(auto timestampOdomPair: mmOdomRGBStampToPose)
                {
                    // calibrate all
                    VectorXd pose = timestampOdomPair.second;
                    VectorXd pose_processed; 
                    if(calibrate)
                        pose_processed = calibratePose(pose);
                    else
                        pose_processed = pose;    

                    mmOdomRGBStampToPose[timestampOdomPair.first] = pose_processed;
                    
                }
                
            }
            
        }

        mbOdomSet = true;
    }

    VectorXd Dataset::calibratePose(VectorXd& pose)
    {
        g2o::SE3Quat pose_c; pose_c.fromVector(pose.tail(7));
        g2o::SE3Quat pose_w = (*mTransGtCalibrate) * pose_c;

        return pose_w.toVector();
    }

    void Dataset::SetCurrentID(int id)
    {
        if ( id >= miTotalNum ){
            std::cout << "Fail. id is larger than totalNum : " << miTotalNum << std::endl;
            return;
        }

        miCurrentID = id;

        return;
            
    }

    double Dataset::GetCurrentTimestamp()
    {
        return GetTimestamp(miCurrentID-1);
    }

    double Dataset::GetTimestamp(int id)
    {
        return stod(mvIdToRGBTimeStamp[id]);
    }
}