cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(libmeshSandbox
  VERSION 0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")

include(FetchContent)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

find_package(MPI REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(libmesh REQUIRED IMPORTED_TARGET libmesh)

# workaround for bogus libmesh include path
get_target_property(
  LIBMESH_INCLUDE_DIRECTORIES PkgConfig::libmesh INTERFACE_INCLUDE_DIRECTORIES)
list(REMOVE_ITEM LIBMESH_INCLUDE_DIRECTORIES "/usr/include/tirpc")
set_property(TARGET PkgConfig::libmesh
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${LIBMESH_INCLUDE_DIRECTORIES})

include(FeatureSummary)
feature_summary(WHAT ALL)

set(SRC_FILES "")

macro(libmesh_sandbox_add_exe folder_name target_name)
  # set_options(EXCLUDE_FROM_ALL)
  # set(one_value_args SOURCES HEADERS)
  # set(multi_value_args EMPTY)
  message(STATUS "adding exe ${target_name} in ${folder_name}")
  add_executable(${folder_name}_${target_name})
  target_sources(${folder_name}_${target_name}
    PRIVATE ${target_name}.cpp
  )
  list(APPEND SRC_FILES "${folder_name}/${target_name}.cpp")
  set(SRC_FILES "${SRC_FILES}" PARENT_SCOPE)
  target_link_libraries(${folder_name}_${target_name}
    PUBLIC
      MPI::MPI_CXX
      PkgConfig::libmesh
      fmt::fmt
  )
  add_test(
    NAME ${folder_name}_${target_name}
    COMMAND ${folder_name}_${target_name}
  )
endmacro()

include(CTest)

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

add_subdirectory(gmsh)

message(STATUS "SRC_FILES = ${SRC_FILES}")

find_program(CLANGFORMAT clang-format)
if (CLANGFORMAT)
  add_custom_target(format
    COMMAND ${CLANGFORMAT}
    -i
    ${SRC_FILES} ${HEADER_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
else()
  add_custom_target(format
    COMMAND echo "clang-format could not be found"
  )
endif()

