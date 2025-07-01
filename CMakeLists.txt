
cmake_minimum_required(VERSION 3.8)

project(NetWork LANGUAGES CXX CUDA)

add_definitions(-w)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) 
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)  

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./include)
include_directories(./include/Layer)
include_directories(.)

aux_source_directory(sources sources_files)
message (${sources_files})


CUDA_ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBS} ${CUDA_cublas_LIBRARY})

target_sources(${PROJECT_NAME} PUBLIC ${sources_files})