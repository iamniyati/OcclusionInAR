cmake_minimum_required(VERSION 3.9)
project(OcclusionInAR)

set(CMAKE_CXX_STANDARD 11)

add_executable(OcclusionInAR main.cpp )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( OcclusionInAR ${OpenCV_LIBS} )
