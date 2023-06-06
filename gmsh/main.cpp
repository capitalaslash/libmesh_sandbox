#include <libmesh/analytic_function.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/equation_systems.h>
#include <libmesh/error_vector.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe.h>
#include <libmesh/kelly_error_estimator.h>
#include <libmesh/libmesh.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/parsed_function.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/sparse_matrix.h>
#include <libmesh/transient_system.h>
#include <libmesh/zero_function.h>

#include <iomanip>
#include <iostream>

using namespace libMesh;

Real f(Point const & /*p*/)
{
  return 0.0;
  // return 2.5*M_PI*M_PI*std::sin(0.5*M_PI*p(0))*std::sin(1.5*M_PI*p(1));
  // return p(0);
  // return M_PI*std::sin(M_PI*p(0));
  // return 2.0;
}

VectorValue<Real> force(Point const & /*p*/)
{
  VectorValue<Real> var;
  var(0) = 0.0;
  var(1) = 0.0;
  var(2) = 0.0;
  return var;
}

void inlet(DenseVector<Real> & output, const Point & /*p*/, const Real /*time*/)
{
  output(0) = 0.;
  output(1) = 1.0; // 4. * p(0) * (1.-p(0));
  output(2) = 0.;
}

void assemble_stokes(EquationSystems & es, const std::string & system_name)
{
  const MeshBase & mesh = es.get_mesh();
  const uint dim = mesh.mesh_dimension();

  TransientLinearImplicitSystem & system =
      es.get_system<TransientLinearImplicitSystem>(system_name);

  const uint u_var = system.variable_number("velx");
  const uint v_var = system.variable_number("vely");
  const uint w_var = system.variable_number("velz");
  const uint p_var = system.variable_number("p");

  FEType fe_type_vel = system.variable_type(u_var);
  std::unique_ptr<FEBase> fe_vel(FEBase::build(dim, fe_type_vel));
  FEType fe_type_p = system.variable_type(p_var);
  std::unique_ptr<FEBase> fe_p(FEBase::build(dim, fe_type_p));

  QGauss qrule(dim, fe_type_vel.default_quadrature_order());
  fe_vel->attach_quadrature_rule(&qrule);
  fe_p->attach_quadrature_rule(&qrule);

  // std::unique_ptr<FEBase> fe_side (FEBase::build(dim, fe_type));
  // QGauss qside (dim-1, FIFTH);
  // fe_side->attach_quadrature_rule (&qside);

  std::vector<Real> const & JxW = fe_vel->get_JxW();
  // std::vector<Real> const& JxW = fe_p->get_JxW();
  std::vector<std::vector<Real>> const & phi = fe_vel->get_phi();
  std::vector<std::vector<RealGradient>> const & dphi = fe_vel->get_dphi();
  std::vector<Point> const & qpoint = fe_vel->get_xyz();
  std::vector<std::vector<Real>> const & psi = fe_p->get_phi();
  // std::vector<std::vector<RealGradient>> const& dpsi = fe_p->get_dphi();

  // const std::vector<std::vector<Real> >& v_side = fe_side->get_phi();
  // const std::vector<Real>& JxW_side = fe_side->get_JxW();
  // const std::vector<Point>& qside_point = fe_side->get_xyz();

  const DofMap & dof_map = system.get_dof_map();

  DenseMatrix<Real> Ke;
  DenseVector<Real> Fe;

  DenseSubMatrix<Real> Kuu(Ke), Kuv(Ke), Kuw(Ke), Kup(Ke), Kvu(Ke), Kvv(Ke), Kvw(Ke),
      Kvp(Ke), Kwu(Ke), Kwv(Ke), Kww(Ke), Kwp(Ke), Kpu(Ke), Kpv(Ke), Kpw(Ke), Kpp(Ke);

  DenseSubVector<Real> Fu(Fe), Fv(Fe), Fw(Fe), Fp(Fe);

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_w;
  std::vector<dof_id_type> dof_indices_p;

  Real const dt = es.parameters.get<Real>("dt");
  Real const nu = 0.1;

  MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for (; el != end_el; ++el)
  {
    const Elem * elem = *el;

    fe_vel->reinit(elem);
    fe_p->reinit(elem);

    dof_map.dof_indices(elem, dof_indices);
    dof_map.dof_indices(elem, dof_indices_u, u_var);
    dof_map.dof_indices(elem, dof_indices_v, v_var);
    dof_map.dof_indices(elem, dof_indices_w, w_var);
    dof_map.dof_indices(elem, dof_indices_p, p_var);

    uint const n_dofs = dof_indices.size();
    uint const n_dofs_vel = dof_indices_u.size();
    uint const n_dofs_p = dof_indices_p.size();

    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);

    Kuu.reposition(u_var * n_dofs_vel, u_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kuv.reposition(u_var * n_dofs_vel, v_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kuw.reposition(u_var * n_dofs_vel, w_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kup.reposition(u_var * n_dofs_vel, p_var * n_dofs_vel, n_dofs_vel, n_dofs_p);

    Kvu.reposition(v_var * n_dofs_vel, u_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kvv.reposition(v_var * n_dofs_vel, v_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kvw.reposition(v_var * n_dofs_vel, w_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kvp.reposition(v_var * n_dofs_vel, p_var * n_dofs_vel, n_dofs_vel, n_dofs_p);

    Kwu.reposition(w_var * n_dofs_vel, u_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kwv.reposition(w_var * n_dofs_vel, v_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kww.reposition(w_var * n_dofs_vel, w_var * n_dofs_vel, n_dofs_vel, n_dofs_vel);
    Kwp.reposition(w_var * n_dofs_vel, p_var * n_dofs_vel, n_dofs_vel, n_dofs_p);

    Kpu.reposition(p_var * n_dofs_vel, u_var * n_dofs_vel, n_dofs_p, n_dofs_vel);
    Kpv.reposition(p_var * n_dofs_vel, v_var * n_dofs_vel, n_dofs_p, n_dofs_vel);
    Kpw.reposition(p_var * n_dofs_vel, w_var * n_dofs_vel, n_dofs_p, n_dofs_vel);
    Kpp.reposition(p_var * n_dofs_vel, p_var * n_dofs_vel, n_dofs_p, n_dofs_p);

    Fu.reposition(u_var * n_dofs_vel, n_dofs_vel);
    Fv.reposition(v_var * n_dofs_vel, n_dofs_vel);
    Fw.reposition(w_var * n_dofs_vel, n_dofs_vel);
    Fp.reposition(p_var * n_dofs_vel, n_dofs_p);

    for (uint qp = 0; qp < qrule.n_points(); qp++)
    {
      VectorValue<Real> vel_old(0.0, 0.0, 0.0);
      VectorValue<Real> gradu_old(0.0, 0.0, 0.0);
      VectorValue<Real> gradv_old(0.0, 0.0, 0.0);
      VectorValue<Real> gradw_old(0.0, 0.0, 0.0);
      for (std::size_t l = 0; l < phi.size(); l++)
      {
        vel_old(0) += phi[l][qp] * system.old_solution(dof_indices_u[l]);
        vel_old(1) += phi[l][qp] * system.old_solution(dof_indices_v[l]);
        vel_old(2) += phi[l][qp] * system.old_solution(dof_indices_w[l]);
        gradu_old.add_scaled(dphi[l][qp], system.old_solution(dof_indices_u[l]));
        gradv_old.add_scaled(dphi[l][qp], system.old_solution(dof_indices_v[l]));
        gradw_old.add_scaled(dphi[l][qp], system.old_solution(dof_indices_w[l]));
      }

      auto f = force(qpoint[qp]);

      for (uint i = 0; i < n_dofs_vel; i++)
      {
        Fu(i) += JxW[qp] * (vel_old(0) * phi[i][qp] + dt * f(0) * phi[i][qp] +
                            dt * (vel_old * gradu_old) * phi[i][qp]);
        Fv(i) += JxW[qp] * (vel_old(1) * phi[i][qp] + dt * f(1) * phi[i][qp] +
                            dt * (vel_old * gradv_old) * phi[i][qp]);
        Fw(i) += JxW[qp] * (vel_old(2) * phi[i][qp] + dt * f(2) * phi[i][qp] +
                            dt * (vel_old * gradw_old) * phi[i][qp]);

        for (uint j = 0; j < n_dofs_vel; j++)
        {
          Kuu(i, j) += JxW[qp] * (phi[j][qp] * phi[i][qp] +
                                  dt * (vel_old * dphi[j][qp]) * phi[i][qp] +
                                  dt * gradu_old(0) * phi[i][qp] * phi[j][qp] +
                                  dt * nu * dphi[j][qp] * dphi[i][qp]);
          Kuv(i, j) += JxW[qp] * dt * gradu_old(1) * phi[i][qp] * phi[j][qp];
          Kuw(i, j) += JxW[qp] * dt * gradu_old(2) * phi[i][qp] * phi[j][qp];

          Kvv(i, j) += JxW[qp] * (phi[j][qp] * phi[i][qp] +
                                  dt * (vel_old * dphi[j][qp]) * phi[i][qp] +
                                  dt * gradv_old(1) * phi[i][qp] * phi[j][qp] +
                                  dt * nu * dphi[j][qp] * dphi[i][qp]);
          Kvu(i, j) += JxW[qp] * dt * gradv_old(0) * phi[i][qp] * phi[j][qp];
          Kvw(i, j) += JxW[qp] * dt * gradv_old(2) * phi[i][qp] * phi[j][qp];

          Kww(i, j) += JxW[qp] * (phi[j][qp] * phi[i][qp] +
                                  dt * (vel_old * dphi[j][qp]) * phi[i][qp] +
                                  dt * gradw_old(2) * phi[i][qp] * phi[j][qp] +
                                  dt * nu * dphi[j][qp] * dphi[i][qp]);
          Kwu(i, j) += JxW[qp] * dt * gradw_old(0) * phi[i][qp] * phi[j][qp];
          Kwv(i, j) += JxW[qp] * dt * gradw_old(1) * phi[i][qp] * phi[j][qp];
        }
        for (uint j = 0; j < n_dofs_p; j++)
        {
          Kup(i, j) += JxW[qp] * (-dt * psi[j][qp] * dphi[i][qp](0));
          Kvp(i, j) += JxW[qp] * (-dt * psi[j][qp] * dphi[i][qp](1));
          Kwp(i, j) += JxW[qp] * (-dt * psi[j][qp] * dphi[i][qp](2));
        }
      }

      for (uint i = 0; i < n_dofs_p; i++)
      {
        for (uint j = 0; j < n_dofs_vel; j++)
        {
          Kpu(i, j) += JxW[qp] * (-dt * dphi[j][qp](0) * psi[i][qp]);
          Kpv(i, j) += JxW[qp] * (-dt * dphi[j][qp](1) * psi[i][qp]);
          Kpw(i, j) += JxW[qp] * (-dt * dphi[j][qp](2) * psi[i][qp]);
        }
        // for (uint j=0; j<n_dofs_p; ++j)
        // {
        //   Kpp(i,j) += JxW[qp]*(dpsi[j][qp]*dpsi[i][qp]);
        // }
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
    //                            Fe(i) += JxW_side[qp] * penalty * value *
    //                            v_side[i][qp]; for (uint j = 0; j < v_side.size();
    //                            j++)
    //                                Ke(i,j) += JxW_side[qp] * penalty * v_side[i][qp]
    //                                * v_side[j][qp];
    //                             //Fe(i) += JxW_side[qp] * 2.0 * v_side[i][qp];
    //                        }

    //                    }
    //                }
    //            }

    // dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);
    dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
    // std::cout << "Ke:\n" << Ke << std::endl;
    // std::cout << "Fe:\n" << Fe << std::endl;

    system.matrix->add_matrix(Ke, dof_indices);
    system.rhs->add_vector(Fe, dof_indices);
  }
}

int main(int argc, char * argv[])
{
  LibMeshInit init(argc, argv);

  Mesh mesh(init.comm());

  mesh.read("cyl_structured.msh");

  mesh.print_info();
  // mesh.boundary_info->print_info();

  EquationSystems es(mesh);

  TransientLinearImplicitSystem & system =
      es.add_system<TransientLinearImplicitSystem>("NS");

  uint u_var = system.add_variable("velx", SECOND, LAGRANGE);
  uint v_var = system.add_variable("vely", SECOND, LAGRANGE);
  uint w_var = system.add_variable("velz", SECOND, LAGRANGE);
  /*uint p_var =*/system.add_variable("p", FIRST, LAGRANGE);

  system.attach_assemble_function(assemble_stokes);
  // system.attach_init_function (init_stokes);

  ZeroFunction<Real> zero;
  ConstFunction<Real> one(1.0);
  // AnalyticFunction<Real> inlet_function(inlet);

  // wall - no slip
  system.get_dof_map().add_dirichlet_boundary(
      DirichletBoundary({3}, {u_var, v_var, w_var}, zero));
  // inlet - imposed velocity
  system.get_dof_map().add_dirichlet_boundary(DirichletBoundary(
      {1}, {w_var}, one)); // ParsedFunction<Number>("16*y*(1-y)*z*(1-z)")));
  system.get_dof_map().add_dirichlet_boundary(
      DirichletBoundary({1}, {u_var, v_var}, zero));

  system.time = 0.0;

  es.init();
  es.print_info();

  Real const dt = 0.5;
  const unsigned int n_timesteps = 10;

  //  const unsigned int n_nonlinear_steps = 15;
  //  const Real nonlinear_tolerance       = 1.e-5;

  es.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
  es.parameters.set<Real>("linear solver tolerance") = 1.e-6;
  es.parameters.set<Real>("dt") = dt;
  es.parameters.set<Real>("nu") = .1;

  auto const filename = "out.e";
  ExodusII_IO exo_io(mesh);
  // exodus requires numeration to start from 1
  exo_io.write_timestep(filename, es, 1, system.time);

  for (uint t_step = 0; t_step < n_timesteps; t_step++)
  {
    system.time += dt;

    libMesh::out << "\nSolving time step " << t_step + 1 << ", time = " << system.time
                 << std::endl;

    *system.old_local_solution = *system.current_local_solution;
    system.solve();

    uint const n_linear_iterations = system.n_linear_iterations();
    Real const final_linear_residual = system.final_linear_residual();

    libMesh::out << "Linear solver converged at step: " << n_linear_iterations
                 << ", final residual: " << final_linear_residual << std::endl;

    exo_io.write_timestep(filename, es, t_step + 2, system.time);
  }

  // system.matrix->close();
  // system.matrix->print();
  // system.rhs->print();
  // system.solution->print();

  //  ErrorVector error;
  //  std::unique_ptr<ErrorEstimator> error_estimator(new KellyErrorEstimator);
  //  error_estimator->estimate_error(system, error);
  //  Real global_error = error.l2_norm();
  //
  //  out << "l2 error = " << global_error << std::endl;

  return 0;
}
