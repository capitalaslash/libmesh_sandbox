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

#include <libmesh/sparse_matrix.h>
#include <libmesh/numeric_vector.h>

#include <libmesh/error_vector.h>
#include <libmesh/kelly_error_estimator.h>

#include <libmesh/exodusII_io.h>

#include <iostream>
#include <iomanip>

using namespace libMesh;

VectorValue<Real> force(Point const & p, Real const /*time*/ = 0.0)
{
  VectorValue<Real> value;
  value(0) = 1.25*M_PI*M_PI * sin(.5*M_PI*p(0)) * sin(M_PI*p(1));
  value(1) = 1.25*M_PI*M_PI * sin(M_PI*p(0)) * sin(.5*M_PI*p(1));
  return value;
}

void assemble_poisson(EquationSystems& es, const std::string& system_name)
{
  const MeshBase& mesh = es.get_mesh();
  const unsigned int dim = mesh.mesh_dimension();

  LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> (system_name);

  const uint u_var = system.variable_number ("u");
  const uint v_var = system.variable_number ("v");

  const DofMap& dof_map = system.get_dof_map();
  FEType fe_type = dof_map.variable_type(0);
  UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
  QGauss qrule (dim, FIFTH);
  fe->attach_quadrature_rule (&qrule);

  UniquePtr<FEBase> fe_side (FEBase::build(dim, fe_type));
  QGauss qside (dim-1, FIFTH);
  fe_side->attach_quadrature_rule (&qside);

  std::vector<Real> const& JxW = fe->get_JxW();

  std::vector<std::vector<Real>> const& phi = fe->get_phi();
  std::vector<std::vector<RealGradient>> const& dphi = fe->get_dphi();
  std::vector<Point> const& qpoint = fe->get_xyz();

  // const std::vector<std::vector<Real> >& v_side = fe_side->get_phi();
  // const std::vector<Real>& JxW_side = fe_side->get_JxW();
  // const std::vector<Point>& qside_point = fe_side->get_xyz();

  DenseMatrix<Real> Ke;
  DenseVector<Real> Fe;

  DenseSubMatrix<Real> Kuu(Ke), Kuv(Ke), Kvu(Ke), Kvv(Ke);
  DenseSubVector<Real> Fu(Fe), Fv(Fe);

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;

  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el ; ++el)
  {
    const Elem* elem = *el;

    dof_map.dof_indices (elem, dof_indices);
    dof_map.dof_indices (elem, dof_indices_u, u_var);
    dof_map.dof_indices (elem, dof_indices_v, v_var);

    fe->reinit (elem);

    uint const n_dofs = dof_indices.size();
    uint const n_dofs_u = dof_indices_u.size();

    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);

    Kuu.reposition (u_var*n_dofs_u, u_var*n_dofs_u, n_dofs_u, n_dofs_u);
    Kuv.reposition (u_var*n_dofs_u, v_var*n_dofs_u, n_dofs_u, n_dofs_u);
    Kvu.reposition (v_var*n_dofs_u, u_var*n_dofs_u, n_dofs_u, n_dofs_u);
    Kvv.reposition (v_var*n_dofs_u, v_var*n_dofs_u, n_dofs_u, n_dofs_u);

    Fu.reposition (u_var*n_dofs_u, n_dofs_u);
    Fv.reposition (v_var*n_dofs_u, n_dofs_u);

    for(uint qp = 0; qp < qrule.n_points(); qp++)
    {
      VectorValue<Real> const f = force(qpoint[qp]);

      for(uint i = 0; i < n_dofs_u; i++)
      {
        Fu(i) += JxW[qp] * f(0) * phi[i][qp];
        Fv(i) += JxW[qp] * f(1) * phi[i][qp];

        for(uint j = 0; j < n_dofs_u; j++)
        {
          Kuu(i,j) += JxW[qp] * (dphi[j][qp] * dphi[i][qp]);
          Kvv(i,j) += JxW[qp] * (dphi[j][qp] * dphi[i][qp]);
        }
      }
    }

    // for (uint side = 0; side<elem->n_sides(); side++)
    // {
    //   if (elem->neighbor(side) == NULL)
    //   {
    //     fe_side->reinit(elem, side);
    //
    //     for (uint qp=0; qp<qside.n_points(); qp++)
    //     {
    //
    //       const Real xf = qside_point[qp](0);
    //       const Real yf = qside_point[qp](1);
    //
    //       const Real penalty = 1.e10;
    //
    //       const Real value = 0.0;
    //
    //       if(yf < 1e-6)
    //       {
    //         for (uint i = 0; i < v_side.size(); i++)
    //         {
    //           Fe(i) += JxW_side[qp] * penalty * value * v_side[i][qp];
    //           for (uint j = 0; j < v_side.size(); j++)
    //             Ke(i,j) += JxW_side[qp] * penalty * v_side[i][qp] * v_side[j][qp];
    //           //Fe(i) += JxW_side[qp] * 2.0 * v_side[i][qp];
    //         }
    //
    //       }
    //     }
    //   }
    // }

    dof_map.heterogenously_constrain_element_matrix_and_vector (Ke, Fe, dof_indices);
    system.matrix->add_matrix (Ke, dof_indices);
    system.rhs->add_vector    (Fe, dof_indices);
  }
}

int main(int argc, char* argv[])
{
  LibMeshInit init(argc, argv);

  Mesh mesh(init.comm());

  MeshTools::Generation::build_square(mesh, 4, 4, 0., 1., 0., 1., QUAD4);

  mesh.print_info();
  // mesh.boundary_info->print_info();

  EquationSystems es(mesh);

  LinearImplicitSystem& system = es.add_system<LinearImplicitSystem>("Diff");

  uint u_var = system.add_variable("u", FIRST, LAGRANGE);
  uint v_var = system.add_variable("v", FIRST, LAGRANGE);

  system.attach_assemble_function(assemble_poisson);

  ZeroFunction<Real> zero;
  // ConstFunction<Real> one(1.0);

  system.get_dof_map().add_dirichlet_boundary(DirichletBoundary({0, 2, 3}, {u_var}, zero));
  system.get_dof_map().add_dirichlet_boundary(DirichletBoundary({0, 1, 3}, {v_var}, zero));

  es.init();

  es.print_info();

  system.solve();

  ErrorVector error;
  UniquePtr<ErrorEstimator> error_estimator(new KellyErrorEstimator);
  error_estimator->estimate_error(system, error);
  Real global_error = error.l2_norm();

  out << "l2 error = " << global_error << std::endl;

  ExodusII_IO(mesh).write_timestep("out.e", es, 1, 0.0);

  return 0;
}

