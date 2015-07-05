cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(openni_grabber)

############## CMAKE OPTIONS #####################

SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/opencv-3.0.0/release)

############### SOURCE FILES ######################

SET(SOURCES openni_grabber.cpp)
add_executable (${PROJECT_NAME} ${SOURCES})

############### MODULES ######################

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries (openni_grabber ${PCL_LIBRARIES} ${OpenCV_LIBS})