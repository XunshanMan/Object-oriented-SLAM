cmake_minimum_required(VERSION 2.8)
project(EllipsoidSLAM)

SET(CMAKE_BUILD_TYPE Release)
# SET(CMAKE_BUILD_TYPE Debug)
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Compile with C14
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# set no warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# For Opencv, check environment first
if(NOT DEFINED ENV{OpenCV_DIR})
   message("not defined environment variable: OpenCV_DIR")
   find_package(OpenCV REQUIRED)
   if(NOT OpenCV_FOUND)
       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
else()
   message("Environment variable OpenCV_DIR: " $ENV{OpenCV_DIR})
   set(OpenCV_DIR $ENV{OpenCV_DIR})
   find_package(OpenCV)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(utils SHARED
        src/utils/matrix_utils.cpp
        src/utils/dataprocess_utils.cpp
        )
target_link_libraries(utils
${OpenCV_LIBS}
)

add_library(${PROJECT_NAME} SHARED
        src/core/Ellipsoid.cpp
        src/core/Map.cpp
        src/core/MapDrawer.cpp
        src/core/Viewer.cpp
        src/core/Initializer.cpp
        src/core/Geometry.cpp
        src/core/System.cpp
        src/core/Tracking.cpp
        src/core/FrameDrawer.cpp
        src/core/Optimizer.cpp
        src/core/Frame.cpp
        src/core/Plane.cpp
        src/core/DataAssociation.cpp
        src/core/BasicEllipsoidEdges.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${Pangolin_LIBRARIES}
        utils
        Config
        symmetry
        EllipsoidExtractor
        PlaneExtractor
        dense_builder
        Polygon
        )

FIND_PACKAGE( PCL REQUIRED )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# add modules
add_subdirectory(src/tum_rgbd)
add_subdirectory(src/dense_builder)
add_subdirectory(src/symmetry)
add_subdirectory(src/config)
add_subdirectory(src/pca)
add_subdirectory(src/plane)
add_subdirectory(src/Polygon)

# interface
add_executable(rgbd
        ./Example/interface/rgbd.cpp
)
target_link_libraries(rgbd
        tum_rgbd
        boost_system
        EllipsoidSLAM
        ${PCL_LIBRARIES}
)       
