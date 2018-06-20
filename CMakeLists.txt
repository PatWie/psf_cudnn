cmake_minimum_required(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(CUDA 9.0 EXACT REQUIRED)
find_package(CUDNN 7 REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(blur_library SHARED
  blur_library.cpp
  blur_library.cu)

# target_link_libraries(blur_library
#   ${PYTHON_LIBRARIES}
#   cudart)

target_link_libraries(blur_library
  ${PYTHON_LIBRARIES} ${CUDNN_LIBRARY})

set_target_properties(blur_library PROPERTIES PREFIX "")
