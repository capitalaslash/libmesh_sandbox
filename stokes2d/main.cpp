#include <libmesh/libmesh.h>

#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>

#include <libmesh/equation_systems.h>
#include <libmesh/linear_implicit_system.h>

#include <libmesh/dof_map.h>
#include <libmesh/fe.h>

#include <libmesh/quadrature_gauss.h>

#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/zero_function.h>
#include <libmesh/analytic_function.h>

#include <libmesh/sparse_matrix.h>
#include <libmesh/numeric_vector.h>

#include <libmesh/error_vector.h>
#include <libmesh/kelly_error_estimator.h>

#include <libmesh/exodusII_io.h>

#include <iostream>
#include <iomanip>

using namespace libMesh;

Real force(Point const& /*p*/)
{
  return 0.0;
  // return 2.5*M_PI*M_PI*std::sin(0.5*M_PI*p(0))*std::sin(1.5*M_PI*p(1));
  // return p(0);
  // return M_PI*std::sin(M_PI*p(0));
  // return 2.0;
}

void inlet_v(DenseVector<Number> & output,
           const Point & p,
           const Real /*time*/)
{
  output(1) = 1.-p(0)*p(0);
}

void assemble_stokes(EquationSystems& es, const std::string& system_name)
{
  const MeshBase& mesh = es.get_mesh();
  const uint dim = mesh.mesh_dimension();

  // LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> (system_name);
  ImplicitSystem& system = es.get_system<ImplicitSystem> (system_name);

  const uint u_var = system.variable_number ("velx");
  const uint v_var = system.variable_number ("vely");
  const uint p_var = system.variable_number ("p");

  const DofMap& dof_map = system.get_dof_map();

  FEType fe_type_vel = dof_map.variable_type(u_var);
  FEType fe_type_p = dof_map.variable_type(p_var);
  UniquePtr<FEBase> fe_vel (FEBase::build(dim, fe_type_vel));
  UniquePtr<FEBase> fe_p (FEBase::build(dim, fe_type_p));
  QGauss qrule (dim, FIFTH);
  fe_vel->attach_quadrature_rule (&qrule);
  fe_p->attach_quadrature_rule (&qrule);

  // UniquePtr<FEBase> fe_side (FEBase::build(dim, fe_type));
  // QGauss qside (dim-1, FIFTH);
  // fe_side->attach_quadrature_rule (&qside);

  std::vector<Real> const& JxW = fe_vel->get_JxW();

  // std::vector<std::vector<Real>> const& phi = fe_vel->get_phi();
  std::vector<std::vector<RealGradient>> const& dphi = fe_vel->get_dphi();
  // std::vector<Point> const& qpoint = fe_vel->get_xyz();
  std::vector<std::vector<Real>> const& psi = fe_p->get_phi();

  // const std::vector<std::vector<Real> >& v_side = fe_side->get_phi();
  // const std::vector<Real>& JxW_side = fe_side->get_JxW();
  // const std::vector<Point>& qside_point = fe_side->get_xyz();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
    Kuu(Ke), Kuv(Ke), Kup(Ke),
    Kvu(Ke), Kvv(Ke), Kvp(Ke),
    Kpu(Ke), Kpv(Ke), Kpp(Ke);

  DenseSubVector<Number>
    Fu(Fe),
    Fv(Fe),
    Fp(Fe);

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_p;

  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el ; ++el)
  {
    const Elem* elem = *el;

    fe_vel->reinit(elem);
    fe_p->reinit(elem);

    dof_map.dof_indices(elem, dof_indices);
    dof_map.dof_indices(elem, dof_indices_u, u_var);
    dof_map.dof_indices(elem, dof_indices_v, v_var);
    dof_map.dof_indices(elem, dof_indices_p, p_var);

    uint const n_dofs = dof_indices.size();
    uint const n_dofs_u = dof_indices_u.size();
    uint const n_dofs_v = dof_indices_v.size();
    uint const n_dofs_p = dof_indices_p.size();

    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);

    Kuu.reposition (u_var*n_dofs_u, u_var*n_dofs_u, n_dofs_u, n_dofs_u);
    Kuv.reposition (u_var*n_dofs_u, v_var*n_dofs_u, n_dofs_u, n_dofs_v);
    Kup.reposition (u_var*n_dofs_u, p_var*n_dofs_u, n_dofs_u, n_dofs_p);

    Kvu.reposition (v_var*n_dofs_v, u_var*n_dofs_v, n_dofs_v, n_dofs_u);
    Kvv.reposition (v_var*n_dofs_v, v_var*n_dofs_v, n_dofs_v, n_dofs_v);
    Kvp.reposition (v_var*n_dofs_v, p_var*n_dofs_v, n_dofs_v, n_dofs_p);

    Kpu.reposition (p_var*n_dofs_u, u_var*n_dofs_u, n_dofs_p, n_dofs_u);
    Kpv.reposition (p_var*n_dofs_u, v_var*n_dofs_u, n_dofs_p, n_dofs_v);
    Kpp.reposition (p_var*n_dofs_u, p_var*n_dofs_u, n_dofs_p, n_dofs_p);

    Fu.reposition (u_var*n_dofs_u, n_dofs_u);
    Fv.reposition (v_var*n_dofs_u, n_dofs_v);
    Fp.reposition (p_var*n_dofs_u, n_dofs_p);

    uint n_qp = qrule.n_points();

    for(uint qp = 0; qp < n_qp; qp++)
    {
      for(uint i = 0; i < n_dofs_u; i++)
      {
        //Fe(i) += JxW[qp] * f * v[i][qp];
        for(uint j = 0; j < n_dofs_u; j++)
        {
          Kuu(i,j) += JxW[qp] * (dphi[j][qp] * dphi[i][qp]);
          Kvv(i,j) += JxW[qp] * (dphi[j][qp] * dphi[i][qp]);
        }
        for (uint j=0; j<n_dofs_p; j++)
        {
          Kup(i,j) += JxW[qp]*(psi[j][qp]*dphi[i][qp](0));
          Kvp(i,j) += JxW[qp]*(psi[j][qp]*dphi[i][qp](1));
        }
      }
      for (uint i=0; i<n_dofs_p; i++)
      {
        for (uint j=0; j<n_dofs_u; j++)
        {
          Kpu(i,j) += JxW[qp]*psi[i][qp]*dphi[j][qp](0);
          Kpv(i,j) += JxW[qp]*psi[i][qp]*dphi[j][qp](1);
        }
      }
    }

    //        for (uint side = 0; side<elem->n_sides(); side++)
    //            if (elem->neighbor(side) == NULL)
    //            {
    //                fe_side->reinit(elem, side);

    //                for (uint qp=0; qp<qside.n_points(); qp++)
    //                {

    //                    const Real xf = qside_point[qp](0);
    //                    const Real yf = qside_point[qp](1);

    //                    const Real penalty = 1.e10;

    //                    const Real value = 0.0;

    //                    if(yf < 1e-6)
    //                    {
    //                        for (uint i = 0; i < v_side.size(); i++)
    //                        {
    //                            Fe(i) += JxW_side[qp] * penalty * value * v_side[i][qp];
    //                            for (uint j = 0; j < v_side.size(); j++)
    //                                Ke(i,j) += JxW_side[qp] * penalty * v_side[i][qp] * v_side[j][qp];
    //                             //Fe(i) += JxW_side[qp] * 2.0 * v_side[i][qp];
    //                        }

    //                    }
    //                }
    //            }

    // dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);
    dof_map.heterogenously_constrain_element_matrix_and_vector (Ke, Fe, dof_indices);
    std::cout << "Ke:\n" << Ke << std::endl;
    std::cout << "Fe:\n" << Fe << std::endl;

    system.matrix->add_matrix (Ke, dof_indices);
    system.rhs->add_vector    (Fe, dof_indices);
  }
}

int main(int argc, char* argv[])
{
  LibMeshInit init(argc, argv);

  Mesh mesh(init.comm());

  const uint nx = (argc == 3)? atoi(argv[1]) : 2;
  const uint ny = (argc == 3)? atoi(argv[2]) : 2;

  MeshTools::Generation::build_square(mesh, nx, ny, -1., 1., -1., 1., QUAD9);

  mesh.print_info();

  // mesh.boundary_info->print_info();

  EquationSystems es(mesh);

  LinearImplicitSystem& system = es.add_system<LinearImplicitSystem>("Diff");
  // ImplicitSystem& system = es.add_system<ImplicitSystem>("Diff");

  uint u_var = system.add_variable("velx", SECOND, LAGRANGE);
  uint v_var = system.add_variable("vely", SECOND, LAGRANGE);
  system.add_variable("p", FIRST, LAGRANGE);

  system.attach_assemble_function(assemble_stokes);

  ZeroFunction<Real> zero;
  ConstFunction<Real> one(1.0);
  // AnalyticFunction<Real> inlet_function_u(inlet_u);
  // AnalyticFunction<Real> inlet_function_v(inlet_v);

  // right side - no slip
  system.get_dof_map().add_dirichlet_boundary(
    DirichletBoundary({1}, {u_var, v_var}, &zero)
  );
  // bottom side - imposed velocity
  system.get_dof_map().add_dirichlet_boundary(
    DirichletBoundary({0}, {v_var}, &one)
  );
  // top side - imposed u
  // system.get_dof_map().add_dirichlet_boundary(
  //   DirichletBoundary({2}, {u_var}, &zero)
  // );
  // left side - symmetry
  system.get_dof_map().add_dirichlet_boundary(
    DirichletBoundary({3}, {u_var}, &zero)
  );

  es.init();

  // es.print_info();

  system.solve();

  system.matrix->close();
  system.matrix->print();
  system.rhs->print();
  system.solution->print();

  //    ErrorVector error;
  //    UniquePtr<ErrorEstimator> error_estimator(new KellyErrorEstimator);
  //    error_estimator->estimate_error(system, error);
  //    Real global_error = error.l2_norm();

  //    out << "l2 error = " << global_error << std::endl;

  std::ostringstream file_name;

  file_name << "out_"
  << std::setw(3)
  << std::setfill('0')
  << std::right
  << 0
  << ".e";

  ExodusII_IO(mesh).write_timestep(file_name.str(), es, 1, 0.0/*system.time*/);

  return 0;
}
