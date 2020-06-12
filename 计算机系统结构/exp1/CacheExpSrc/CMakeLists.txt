cmake_minimum_required(VERSION 3.5)
project(CacheSim)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")


#aux_source_directory(. SOURCE_FILES)
#add_executable(CacheSim ${SOURCE_FILES})
add_executable(CacheSim main.cpp CacheSim.cpp)