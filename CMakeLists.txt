cmake_minimum_required(VERSION 3.9)
project(Histogram)

set(CMAKE_CXX_STANDARD 11)

add_executable(Histogram main.cpp )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( Histogram ${OpenCV_LIBS} )