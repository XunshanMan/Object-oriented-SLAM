# set(CMAKE_BUILD_TYPE Debug)
add_library(tum_rgbd SHARED
    io.cpp
)

target_link_libraries(tum_rgbd
        ${OpenCV_LIBS}
        utils
        )
