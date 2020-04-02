// A parameter file supporting global dynamic changes

#ifndef ELLIPSOIDSLAM_CONFIG_H
#define ELLIPSOIDSLAM_CONFIG_H

#include <opencv2/core/core.hpp>
#include <memory>
#include <iostream>
#include <map>

using namespace std;

namespace EllipsoidSLAM {
    class Config{
    public:
        static void SetParameterFile( const string& filename );

        // Get function reads value from parameter file (.yaml)
        template <typename T>
        static T Get(const string& key){
            return T(Config::mConfig->mFile[key]);
        }

        template <typename T>
        static void Set(const string& key, T value){
            Config::mConfig->mFile << key << value;    // tobe test
        }

        ~Config();

        static void Init();

        // set the parameter value, if do not exist, create a new one
        template <typename T>
        static void SetValue(const string& key, T value)
        {
            auto pos = Config::mConfig->mKeyValueMap.find(key);
            if( pos == Config::mConfig->mKeyValueMap.end() )  // do not exist
            {
                Config::mConfig->mKeyValueMap.insert(make_pair(key, double(value)));
            }
            else
            {
                pos->second = double(value);
            }            
        }

        // ReadValue function reads parameter value set by SetValue function, and return default value if it does not exist
        template <typename T>
        static T ReadValue(const string& key, T default_value = 0)
        {
            auto pos = Config::mConfig->mKeyValueMap.find(key);
            if( pos == Config::mConfig->mKeyValueMap.end() )  // do not exist
            {
                return Get<T>(key);    // return value in the parameter file
            }
            else
            {
                return T(pos->second);
            }            
        }

    private:
        Config(){};
        static std::shared_ptr<Config> mConfig;
        cv::FileStorage mFile;

        std::map<string,double> mKeyValueMap;

    };
}
#endif //ELLIPSOIDSLAM_CONFIG_H