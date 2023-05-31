libmesh_sandbox_add_exe(gmsh main)

file(COPY cyl_structured.msh
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
