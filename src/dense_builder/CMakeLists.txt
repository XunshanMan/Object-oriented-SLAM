add_library(dense_builder SHARED
    builder.cpp
)

target_link_libraries(dense_builder
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        utils
        )
