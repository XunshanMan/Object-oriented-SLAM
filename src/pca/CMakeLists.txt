# set(CMAKE_BUILD_TYPE Debug)

add_library(EllipsoidExtractor SHARED
    EllipsoidExtractor.cpp
)

target_link_libraries(EllipsoidExtractor
        ${OpenCV_LIBS}
        symmetry
        utils
        )
