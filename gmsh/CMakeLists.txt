add_executable(gmsh main.cxx)

file(COPY cyl_structured.msh
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
