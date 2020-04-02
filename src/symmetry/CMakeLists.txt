# set(CMAKE_BUILD_TYPE Debug)

add_library(symmetry SHARED
    Symmetry.cpp
    SymmetrySolver.cpp
    BorderExtractor.cpp
    PointCloudFilter.cpp
)

message ("source dir of symmetry: " ${PROJECT_SOURCE_DIR})
target_link_libraries(symmetry
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        utils
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )
