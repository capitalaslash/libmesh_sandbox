#include <libmesh/boundary_info.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>

// Bring in the libmesh namespace
using namespace libMesh;

int main(int argc, char * argv[])
{
  LibMeshInit init(argc, argv);

  Mesh msh(init.comm(), 1);
  EquationSystems es(msh);

  es.add_system<System>("a_system");
  es.get_system<System>("a_system").add_variable("a_variable");

  MeshTools::Generation::build_line(msh, 10, 0., 1., EDGE2);
  msh.prepare_for_use();
  msh.print_info();
  msh.boundary_info->print_info();

  es.init();
  es.print_info();

  std::unique_ptr<Elem> e = Elem::build(EDGE2);
  e->set_id(msh.max_elem_id());
  out << msh.max_elem_id() << std::endl;
  e->set_node(0) = msh.node_ptr(2);
  e->set_node(1) = msh.node_ptr(8);

  msh.add_elem(e.get());
  msh.prepare_for_use();
  msh.print_info();
  msh.boundary_info->print_info();

  es.reinit();
  es.print_info();

  return 0;
}
