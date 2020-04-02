
add_library(Config SHARED
    Config.cpp
)

target_link_libraries(Config
        ${OpenCV_LIBS}
        )
