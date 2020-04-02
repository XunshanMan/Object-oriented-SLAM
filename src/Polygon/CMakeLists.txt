add_library(Polygon SHARED
 Polygon.cpp)
target_link_libraries(Polygon ${OpenCV_LIBS})
