#ifndef ELLIPSOIDSLAM_DATAPROCESS_UTILS_H
#define ELLIPSOIDSLAM_DATAPROCESS_UTILS_H

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

// read data from a txt file and store as an eigen matrix
Eigen::MatrixXd readDataFromFile(const char* fileName, bool dropFirstline = false);

std::vector<std::vector<std::string>> readStringFromFile(const char* fileName, int dropLines = 0);

// save eigen matrix data to a text file
bool saveMatToFile(Eigen::MatrixXd &matIn, const char* fileName);

// get all the file names under a dir. 
void GetFileNamesUnderDir(string path,vector<string>& filenames);
string splitFileNameFromFullDir(string &s, bool bare = false);

// sort file names in ascending order 
void sortFileNames(vector<string>& filenames, vector<string>& filenamesSorted);

bool calibrateMeasurement(Eigen::Vector4d &measure , int rows, int cols, int config_boarder = 10, int config_size = 100);

#endif //ELLIPSOIDSLAM_DATAPROCESS_UTILS_H
