cmake_minimum_required(VERSION 2.8)

project(libmeshSandbox CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wpedantic -std=c++11")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

find_package(LIBMESH REQUIRED)
add_definitions(${LIBMESH_DEFINITIONS})
include_directories(${LIBMESH_INCLUDE_DIRS})
set(LIBS ${LIBS} ${LIBMESH_LIBRARIES})

add_subdirectory(hermite1d)

add_subdirectory(elem_add)

add_subdirectory(self_lumping1d)

