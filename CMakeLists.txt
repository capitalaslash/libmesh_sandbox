cmake_minimum_required(VERSION 2.8)

project(libmeshSandbox CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wpedantic")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

find_package(LIBMESH REQUIRED)
add_definitions(${LIBMESH_DEFINITIONS})
include_directories(${LIBMESH_INCLUDE_DIRS})
set(LIBS ${LIBS} ${LIBMESH_LIBRARIES})
link_libraries(${LIBS})

add_subdirectory(test1d)

add_subdirectory(advection1d)

add_subdirectory(test2d)

add_subdirectory(dirichlet)

add_subdirectory(hermite1d)

add_subdirectory(elem_add)

add_subdirectory(self_lumping1d)

add_subdirectory(stokes2d)
add_subdirectory(stokes3d)

add_subdirectory(neumann2d)

