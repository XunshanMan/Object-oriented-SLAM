

# SET(CMAKE_BUILD_TYPE Debug)

add_library(PlaneExtractor SHARED
PlaneExtractor.cpp
)

target_link_libraries(PlaneExtractor
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Config
        )
