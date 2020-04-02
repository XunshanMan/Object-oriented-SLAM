
#include "Config.h"

namespace EllipsoidSLAM{
    void Config::SetParameterFile( const std::string& filename )
    {
        if ( mConfig == nullptr )
            mConfig = shared_ptr<Config>(new Config);
        mConfig->mFile = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );
        if ( !mConfig->mFile.isOpened())
        {
            std::cerr<<"parameter file "<< filename <<" does not exist."<<std::endl;
            mConfig->mFile.release();
            return;
        }
    }

    Config::~Config()
    {
        if ( mFile.isOpened() )
            mFile.release();
    }

    void Config::Init(){
        if ( mConfig == nullptr ){
            mConfig = shared_ptr<Config>(new Config);

            //  ************ default parameters here *************
            mConfig->SetValue("Tracking_MINIMUM_INITIALIZATION_FRAME", 15);       // The minimum frame number required for ellipsoid initialization using 2d observations.
            mConfig->SetValue("EllipsoidExtractor_DEPTH_RANGE", 6);         // the valid depth range (m)

        }
    }

    shared_ptr<Config> Config::mConfig = nullptr;
}