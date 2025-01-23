cmake_minimum_required(VERSION 3.19)
project(FPS_CPU)

set(CMAKE_CXX_STANDARD 11)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  src/Interval.cpp
  src/KDLineTree.cpp
  src/KDNode.cpp
  src/KDTree.cpp
  src/KDTreeBase.cpp
  src/Point.cpp
)