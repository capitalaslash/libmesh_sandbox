#include <libmesh/analytic_function.h>
#include <libmesh/boundary_info.h>
#include <libmesh/const_function.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/dense_submatrix.h>
#include <libmesh/dense_subvector.h>
#include <libmesh/dense_vector.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/elem.h>
#include <libmesh/enum_solver_package.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe.h>
#include <libmesh/getpot.h>
#include <libmesh/libmesh.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/perf_log.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/sparse_matrix.h>
#include <libmesh/transient_system.h>
#include <libmesh/utility.h>
#include <libmesh/zero_function.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <array>
#include <functional>

using namespace libMesh;

void assemble_ns(EquationSystems & es, std::string const & system_name);
void assemble_t(EquationSystems & es, std::string const & system_name);

void set_lid_driven_bcs(TransientLinearImplicitSystem & system);
void set_stagnation_bcs(TransientLinearImplicitSystem & system);
void set_poiseuille_bcs(TransientLinearImplicitSystem & system);

// The main program.
int main(int argc, char ** argv)
{
  LibMeshInit init(argc, argv);

  auto const sep = std::string(88, '=') + "\n";

  GetPot command_line(argc, argv);

  int n_elem = 20;
  if (command_line.search(1, "-n_elem"))
    n_elem = command_line.next(n_elem);

  Mesh mesh(init.comm());
  MeshTools::Generation::build_square(mesh, n_elem, n_elem, 0., 1., 0., 1., QUAD9);
  mesh.print_info();

  EquationSystems equation_systems(mesh);

  auto & system_ns =
      equation_systems.add_system<TransientLinearImplicitSystem>("Navier-Stokes");
  system_ns.add_variable("vel_x", SECOND);
  system_ns.add_variable("vel_y", SECOND);
  system_ns.add_variable("p", FIRST);
  system_ns.attach_assemble_function(assemble_ns);

  // Note: only pick one set of BCs!
  // set_lid_driven_bcs(system);
  // set_stagnation_bcs(system);
  set_poiseuille_bcs(system_ns);

  auto & system_t =
      equation_systems.add_system<TransientLinearImplicitSystem>("Temperature");
  system_t.add_variable("T", FIRST);
  system_t.attach_assemble_function(assemble_t);

  DofMap & dof_map_t = system_t.get_dof_map();

  // T=0 on bottom, left, right
  dof_map_t.add_dirichlet_boundary(
      DirichletBoundary({0, 1, 3}, {0}, ZeroFunction<Number>()));
  // T=1 on top
  dof_map_t.add_dirichlet_boundary(
      DirichletBoundary({2}, {0}, ConstFunction<Number>(1.)));

  equation_systems.init();
  equation_systems.print_info();

  PerfLog perf_log_ns("Navier-Stokes");

  Real const dt = 0.1;
  Real const t_init = 0.0;
  system_ns.time = t_init;
  system_t.time = t_init;
  uint const n_timesteps = 15;

  uint const n_nonlinear_steps = 15;
  Real const nonlinear_tolerance = 1.e-5;
  Real const initial_linear_solver_tol = 1.e-6;

  uint const write_interval = 1;

  int max_iter = 25;
  if (command_line.search(1, "-max_iter"))
    max_iter = command_line.next(max_iter);

  equation_systems.parameters.set<uint>("linear solver maximum iterations") = max_iter;
  equation_systems.parameters.set<Real>("dt") = dt;
  equation_systems.parameters.set<Real>("nu") = .01;

  std::unique_ptr<NumericVector<Number>> last_nonlinear_soln(
      system_ns.solution->clone());

  // Since we are not doing adaptivity, write all solutions to a single Exodus file.
  ExodusII_IO exo_io(mesh);

  for (uint t_step = 1; t_step <= n_timesteps; ++t_step)
  {
    system_ns.time += dt;
    system_t.time += dt;

    fmt::print(out, "\n{}time step: {}, time: {}\n", sep, t_step, system_ns.time);

    *system_ns.old_local_solution = *system_ns.current_local_solution;
    *system_t.old_local_solution = *system_t.current_local_solution;

    equation_systems.parameters.set<Real>("linear solver tolerance") =
        initial_linear_solver_tol;

    bool converged = false;
    for (uint l = 0; l < n_nonlinear_steps; ++l)
    {
      last_nonlinear_soln->zero();
      last_nonlinear_soln->add(*system_ns.solution);

      perf_log_ns.push("linear solve");
      system_ns.solve();
      perf_log_ns.pop("linear solve");

      last_nonlinear_soln->add(-1., *system_ns.solution);
      last_nonlinear_soln->close();
      const Real norm_delta = last_nonlinear_soln->l2_norm();
      const uint n_linear_iterations = system_ns.n_linear_iterations();
      const Real final_linear_residual = system_ns.final_linear_residual();

      if (n_linear_iterations == 0 &&
          (system_ns.final_linear_residual() >= nonlinear_tolerance || l == 0))
      {
        Real old_linear_solver_tolerance =
            equation_systems.parameters.get<Real>("linear solver tolerance");
        equation_systems.parameters.set<Real>("linear solver tolerance") =
            1.e-3 * old_linear_solver_tolerance;
        continue;
      }

      fmt::print(out, "ns linear solver converged at step: {}\n", n_linear_iterations);
      fmt::print(out, "ns final residual: {:e}\n", final_linear_residual);
      fmt::print(out, "ns nonlinear convergence: ||u - u_old||: {:e}\n", norm_delta);

      if ((norm_delta < nonlinear_tolerance) &&
          (system_ns.final_linear_residual() < nonlinear_tolerance))
      {
        fmt::print(out, "nonlinear solver converged at step {}\n", l);
        converged = true;
        break;
      }

      Real const new_linear_solver_tolerance =
          std::min(Utility::pow<2>(final_linear_residual), initial_linear_solver_tol);
      equation_systems.parameters.set<Real>("linear solver tolerance") =
          new_linear_solver_tolerance;
    } // end nonlinear loop

    libmesh_error_msg_if(!converged, "Error: Newton iterations failed to converge!");

    system_t.solve();
    fmt::print(
        out, "T linear solver converged at step: {}\n", system_t.n_linear_iterations());
    fmt::print(out, "T final residual: {:e}\n", system_t.final_linear_residual());

    // write out every nth timestep to file.
    if ((t_step + 1) % write_interval == 0)
    {
      exo_io.write_timestep("out.e", equation_systems, t_step + 1, system_ns.time);
    }
  } // end timestep loop.

  return 0;
}

void assemble_ns(EquationSystems & es, const std::string & libmesh_dbg_var(system_name))
{
  libmesh_assert_equal_to(system_name, "Navier-Stokes");

  MeshBase const & mesh = es.get_mesh();
  uint const dim = mesh.mesh_dimension();

  auto & system_ns = es.get_system<TransientLinearImplicitSystem>("Navier-Stokes");
  uint const u_var = system_ns.variable_number("vel_x");
  uint const v_var = system_ns.variable_number("vel_y");
  uint const p_var = system_ns.variable_number("p");

  FEType fe_vel_type = system_ns.variable_type(u_var);
  std::unique_ptr<FEBase> fe_vel(FEBase::build(dim, fe_vel_type));

  FEType fe_pres_type = system_ns.variable_type(p_var);
  std::unique_ptr<FEBase> fe_pres(FEBase::build(dim, fe_pres_type));

  QGauss qrule(dim, fe_vel_type.default_quadrature_order());
  fe_vel->attach_quadrature_rule(&qrule);
  fe_pres->attach_quadrature_rule(&qrule);

  std::vector<Real> const & JxW = fe_vel->get_JxW();
  std::vector<std::vector<Real>> const & v = fe_vel->get_phi();
  std::vector<std::vector<RealGradient>> const & dv = fe_vel->get_dphi();
  std::vector<std::vector<Real>> const & q = fe_pres->get_phi();

  DofMap const & dof_map = system_ns.get_dof_map();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;
  DenseSubMatrix<Number> Kuu(Ke), Kuv(Ke), Kup(Ke);
  DenseSubMatrix<Number> Kvu(Ke), Kvv(Ke), Kvp(Ke);
  DenseSubMatrix<Number> Kpu(Ke), Kpv(Ke), Kpp(Ke);
  DenseSubVector<Number> Fu(Fe), Fv(Fe), Fp(Fe);

  std::reference_wrapper<DenseSubVector<Number>> F[2] = {Fu, Fv};
  std::reference_wrapper<DenseSubMatrix<Number>> K[2][2] = {{Kuu, Kuv}, {Kvu, Kvv}};
  std::reference_wrapper<DenseSubMatrix<Number>> B[2] = {Kup, Kvp};
  std::reference_wrapper<DenseSubMatrix<Number>> BT[2] = {Kpu, Kpv};

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_p;

  Real const dt = es.parameters.get<Real>("dt");
  Real const theta = 1.;
  Real const nu = es.parameters.get<Real>("nu");
  bool const pin_pressure = es.parameters.get<bool>("pin_pressure");

  for (const auto & elem: mesh.active_local_element_ptr_range())
  {
    dof_map.dof_indices(elem, dof_indices);
    dof_map.dof_indices(elem, dof_indices_u, u_var);
    dof_map.dof_indices(elem, dof_indices_v, v_var);
    dof_map.dof_indices(elem, dof_indices_p, p_var);

    uint const n_dofs = dof_indices.size();
    uint const n_u_dofs = dof_indices_u.size();
    uint const n_v_dofs = dof_indices_v.size();
    uint const n_p_dofs = dof_indices_p.size();

    fe_vel->reinit(elem);
    fe_pres->reinit(elem);

    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);
    Kuu.reposition(u_var * n_u_dofs, u_var * n_u_dofs, n_u_dofs, n_u_dofs);
    Kuv.reposition(u_var * n_u_dofs, v_var * n_u_dofs, n_u_dofs, n_v_dofs);
    Kup.reposition(u_var * n_u_dofs, p_var * n_u_dofs, n_u_dofs, n_p_dofs);
    Kvu.reposition(v_var * n_v_dofs, u_var * n_v_dofs, n_v_dofs, n_u_dofs);
    Kvv.reposition(v_var * n_v_dofs, v_var * n_v_dofs, n_v_dofs, n_v_dofs);
    Kvp.reposition(v_var * n_v_dofs, p_var * n_v_dofs, n_v_dofs, n_p_dofs);
    Kpu.reposition(p_var * n_u_dofs, u_var * n_u_dofs, n_p_dofs, n_u_dofs);
    Kpv.reposition(p_var * n_u_dofs, v_var * n_u_dofs, n_p_dofs, n_v_dofs);
    Kpp.reposition(p_var * n_u_dofs, p_var * n_u_dofs, n_p_dofs, n_p_dofs);
    Fu.reposition(u_var * n_u_dofs, n_u_dofs);
    Fv.reposition(v_var * n_u_dofs, n_v_dofs);
    Fp.reposition(p_var * n_u_dofs, n_p_dofs);

    for (uint qp = 0; qp < qrule.n_points(); qp++)
    {
      NumberVectorValue u_old;
      NumberVectorValue u;
      Number p_old = 0.;
      std::array<Gradient, 2> grad_u{};
      std::array<Gradient, 2> grad_u_old{};
      for (uint l = 0; l < n_u_dofs; l++)
      {
        u_old(0) += v[l][qp] * system_ns.old_solution(dof_indices_u[l]);
        u_old(1) += v[l][qp] * system_ns.old_solution(dof_indices_v[l]);
        grad_u_old[0].add_scaled(dv[l][qp], system_ns.old_solution(dof_indices_u[l]));
        grad_u_old[1].add_scaled(dv[l][qp], system_ns.old_solution(dof_indices_v[l]));

        u(0) += v[l][qp] * system_ns.current_solution(dof_indices_u[l]);
        u(1) += v[l][qp] * system_ns.current_solution(dof_indices_v[l]);
        grad_u[0].add_scaled(dv[l][qp], system_ns.current_solution(dof_indices_u[l]));
        grad_u[1].add_scaled(dv[l][qp], system_ns.current_solution(dof_indices_v[l]));
      }

      for (uint l = 0; l < n_p_dofs; l++)
      {
        p_old += q[l][qp] * system_ns.old_solution(dof_indices_p[l]);
      }

      for (uint i = 0; i < n_u_dofs; i++)
      {
        for (uint k = 0; k < 2; ++k)
          F[k](i) += JxW[qp] *
                     (u_old(k) * v[i][qp] - // mass-matrix term
                      (1. - theta) * dt * (u_old * grad_u_old[k]) *
                          v[i][qp] +                             // convection term
                      (1. - theta) * dt * p_old * dv[i][qp](k) - // pressure term on rhs
                      (1. - theta) * dt * nu *
                          (grad_u_old[k] * dv[i][qp]) +         // diffusion term on rhs
                      theta * dt * (u * grad_u[k]) * v[i][qp]); // Newton term

        for (uint j = 0; j < n_u_dofs; j++)
          for (uint k = 0; k < 2; ++k)
            for (uint l = 0; l < 2; ++l)
            {
              if (k == l)
                K[k][k](i, j) +=
                    JxW[qp] *
                    (v[i][qp] * v[j][qp] +                       // mass matrix term
                     theta * dt * nu * (dv[i][qp] * dv[j][qp]) + // diffusion term
                     theta * dt * (u * dv[j][qp]) * v[i][qp]);   // convection term

              K[k][l](i, j) +=
                  JxW[qp] * theta * dt * grad_u[k](l) * v[i][qp] * v[j][qp];
            }

        for (uint j = 0; j < n_p_dofs; j++)
          for (uint k = 0; k < 2; ++k)
            B[k](i, j) += JxW[qp] * -theta * dt * q[j][qp] * dv[i][qp](k);
      }

      for (uint i = 0; i < n_p_dofs; i++)
        for (uint j = 0; j < n_u_dofs; j++)
          for (uint k = 0; k < 2; ++k)
            BT[k](i, j) -= JxW[qp] * q[i][qp] * dv[j][qp](k);
    } // end of the quadrature point qp-loop

    if (pin_pressure)
    {
      Real const penalty = 1.e10;
      uint const pressure_node = 0;
      Real const p_value = 0.0;
      for (auto c: elem->node_index_range())
        if (elem->node_id(c) == pressure_node)
        {
          Kpp(c, c) += penalty;
          Fp(c) += penalty * p_value;
        }
    }

    dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

    system_ns.get_system_matrix().add_matrix(Ke, dof_indices);
    system_ns.rhs->add_vector(Fe, dof_indices);
  } // end of element loop
}

void set_lid_driven_bcs(TransientLinearImplicitSystem & system_ns)
{
  system_ns.get_equation_systems().parameters.set<bool>("pin_pressure") = true;

  uint const u_var = system_ns.variable_number("vel_x");
  uint const v_var = system_ns.variable_number("vel_y");

  DofMap & dof_map = system_ns.get_dof_map();

  // u=v=0 on bottom, left, right
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({0, 1, 3}, {u_var, v_var}, ZeroFunction<Number>()));
  // u=1 on top
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({2}, {u_var}, ConstFunction<Number>(1.)));
  // v=0 on top
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({2}, {v_var}, ZeroFunction<Number>()));
}

void top_x(DenseVector<Number> & output, Point const & p, Real const /*t*/)
{
  output(0) = p(0);
};

void top_y(DenseVector<Number> & output, Point const & p, Real const /*t*/)
{
  output(0) = -p(1);
};

void set_stagnation_bcs(TransientLinearImplicitSystem & system_ns)
{
  system_ns.get_equation_systems().parameters.set<bool>("pin_pressure") = false;

  uint const u_var = system_ns.variable_number("vel_x");
  uint const v_var = system_ns.variable_number("vel_y");

  DofMap & dof_map = system_ns.get_dof_map();

  // u=v=0 on bottom (boundary 0)
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({0}, {u_var, v_var}, ZeroFunction<Number>()));
  // u=0 on left (boundary 3) (symmetry)
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({3}, {u_var}, ZeroFunction<Number>()));
  {
    // u = k*x on top (boundary 2)
    // auto const additional_vars = std::vector<std::string>({"k"});
    // auto const initial_vals = std::vector<Number>{1.};
    dof_map.add_dirichlet_boundary(DirichletBoundary(
        {2},
        {u_var},
        // ParsedFunction<Number>("k*x", &additional_vars, &initial_vals)
        AnalyticFunction<Number>(top_x)));
  }
  {
    // v = -k*y on top (boundary 2)
    // std::vector<std::string> additional_vars{"k"};
    // std::vector<Number> initial_vals{1.};
    dof_map.add_dirichlet_boundary(DirichletBoundary(
        {2},
        {v_var},
        // ParsedFunction<Number>("-k*y", &additional_vars, &initial_vals),
        AnalyticFunction<Number>(top_y),
        LOCAL_VARIABLE_ORDER));
  }
}

void inlet(DenseVector<Number> & output, Point const & p, Real const /*t*/)
{
  output(0) = 4.0 * p(1) * (1.0 - p(1));
};

void set_poiseuille_bcs(TransientLinearImplicitSystem & system_ns)
{
  system_ns.get_equation_systems().parameters.set<bool>("pin_pressure") = false;

  uint const u_var = system_ns.variable_number("vel_x");
  uint const v_var = system_ns.variable_number("vel_y");

  DofMap & dof_map = system_ns.get_dof_map();

  // u=v=0 on top, bottom
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({0, 2}, {u_var, v_var}, ZeroFunction<Number>()));
  // u=quadratic on left
  // dof_map.add_dirichlet_boundary(
  //     DirichletBoundary({3}, {u_var}, ParsedFunction<Number>("4*y*(1-y)")));
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({3}, {u_var}, AnalyticFunction<Number>(inlet)));
  // v=0 on left
  dof_map.add_dirichlet_boundary(
      DirichletBoundary({3}, {v_var}, ZeroFunction<Number>()));
}

void assemble_t(EquationSystems & es, std::string const & system_name)
{
  libmesh_assert_equal_to(system_name, "Temperature");

  MeshBase const & mesh = es.get_mesh();
  uint const dim = mesh.mesh_dimension();

  auto & system_t = es.get_system<TransientLinearImplicitSystem>("Temperature");
  uint const t_var = system_t.variable_number("T");

  auto & system_ns = es.get_system<TransientLinearImplicitSystem>("Navier-Stokes");
  uint const u_var = system_ns.variable_number("vel_x");
  uint const v_var = system_ns.variable_number("vel_y");

  FEType fe_t_type = system_t.variable_type(t_var);
  std::unique_ptr<FEBase> fe_t(FEBase::build(dim, fe_t_type));

  FEType fe_vel_type = system_ns.variable_type(u_var);
  std::unique_ptr<FEBase> fe_vel(FEBase::build(dim, fe_vel_type));

  QGauss qrule(dim, fe_vel_type.default_quadrature_order());
  fe_t->attach_quadrature_rule(&qrule);
  fe_vel->attach_quadrature_rule(&qrule);

  std::vector<Real> const & JxW = fe_t->get_JxW();
  std::vector<std::vector<Real>> const & q = fe_t->get_phi();
  std::vector<std::vector<RealGradient>> const & dq = fe_t->get_dphi();
  std::vector<std::vector<Real>> const & v = fe_vel->get_phi();

  DofMap const & dof_map_t = system_t.get_dof_map();
  DofMap const & dof_map_ns = system_ns.get_dof_map();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices_t;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;

  Real const dt = es.parameters.get<Real>("dt");
  // Real const theta = 1.;
  Real const kappa = 0.1; // es.parameters.get<Real>("kappa");

  for (const auto & elem: mesh.active_local_element_ptr_range())
  {
    dof_map_t.dof_indices(elem, dof_indices_t);
    uint const n_dofs_t = dof_indices_t.size();

    dof_map_ns.dof_indices(elem, dof_indices_u, u_var);
    dof_map_ns.dof_indices(elem, dof_indices_v, v_var);
    uint const n_dofs_vel = dof_indices_u.size();

    fe_t->reinit(elem);
    fe_vel->reinit(elem);

    Ke.resize(n_dofs_t, n_dofs_t);
    Fe.resize(n_dofs_t);

    for (uint qp = 0; qp < qrule.n_points(); qp++)
    {
      Number t_old = 0.;
      for (uint l = 0; l < n_dofs_t; l++)
      {
        t_old += q[l][qp] * system_t.old_solution(dof_indices_t[l]);
      }

      NumberVectorValue u_old;
      for (uint l = 0; l < n_dofs_vel; l++)
      {
        u_old(0) += v[l][qp] * system_ns.old_solution(dof_indices_u[l]);
        u_old(1) += v[l][qp] * system_ns.old_solution(dof_indices_v[l]);
      }

      for (uint i = 0; i < n_dofs_t; i++)
      {
        Fe(i) += JxW[qp] *
                 (t_old * q[i][qp] // mass-matrix term
                                   // - (1. - theta) * dt * (u_old * grad_u_old[k]) *
                  //     v[i][qp] +                             // convection term
                  // (1. - theta) * dt * p_old * dv[i][qp](k) - // pressure term on rhs
                  // (1. - theta) * dt * nu *
                  //     (grad_u_old[k] * dv[i][qp]) +         // diffusion term on rhs
                  // theta * dt * (u * grad_u[k]) * v[i][qp] // Newton term
                 );

        for (uint j = 0; j < n_dofs_t; j++)
        {
          Ke(i, j) +=
              JxW[qp] *
              (q[i][qp] * q[j][qp]                               // mass matrix term
               + dt * kappa * (dq[i][qp] * dq[j][qp])            // diffusion term
               + /*theta **/ dt * (u_old * dq[j][qp]) * q[i][qp] // convection term
              );
        } // j loop
      }   // i loop
    }     // end of the quadrature point qp-loop
    dof_map_t.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices_t);

    system_t.get_system_matrix().add_matrix(Ke, dof_indices_t);
    system_t.rhs->add_vector(Fe, dof_indices_t);
  } // end of element loop
}
