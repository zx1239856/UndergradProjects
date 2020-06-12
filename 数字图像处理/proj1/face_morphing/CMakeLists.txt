cmake_minimum_required(VERSION 3.08)
project(FaceMorphing)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS "-mavx2 -msse4")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(Eigen3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
set(target_lib ${OpenCV_LIBS} glog::glog CGAL::CGAL_Core)

add_executable(face main.cpp)
target_link_libraries(face ${target_lib})
