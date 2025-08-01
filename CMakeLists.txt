cmake_minimum_required(VERSION 3.18)
project(PRENOM)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()


add_definitions(-Wno-unused-value -Wno-unknown-pragmas -Wno-sequence-point 
               -Wno-delete-non-virtual-dtor -Wno-unused-but-set-variable 
               -Wno-sign-compare -Wno-unused-variable -Wno-return-local-addr 
               -Wno-unused-function -Wno-deprecated-declarations -msse4.2 -mfpmath=sse)


MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


###########
# OpenMP  #
###########
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

##########
# OpenCV #
##########
find_package(OpenCV 4.2 QUIET)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV >= 4.2 not found.")
endif()

#########
# Eigen #
#########
find_package(Eigen3 REQUIRED)

############
# Pangolin #
############
find_package(Pangolin REQUIRED)

###########
# PCL     #
###########
find_package(PCL REQUIRED)

##################
# Mutil_Object_NeRF #
##################
include_directories(${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core)
include_directories(${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core/include)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # to tell your code it was found via #ifdef USE_CUDA:
    add_definitions(-DUSE_CUDA)    

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${CUDA_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/third-party
${PROJECT_SOURCE_DIR}/third-party/farthest_point_sampling/include
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/ObjectFrame.cc
src/OptimizeObject.cc
src/ObjectMap.cc
src/Utils.cc
src/ObjectManager.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${PROJECT_SOURCE_DIR}/third-party/farthest_point_sampling/lib/libFPS_CPU.so
${PROJECT_SOURCE_DIR}/dependencies/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/dependencies/line_lbd/lib/libline_lbd.so
${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core/build/libMON.so
)

# Build examples - Monocular
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/Monocular)
add_executable(mono_tum
examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

# Build examples - RGBD
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/RGBD)
add_executable(rgbd_tum
examples/RGBD/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;90;120")
