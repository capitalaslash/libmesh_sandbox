// The libMesh Finite Element Library.
// Copyright (C) 2002-2017 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



// <h1>Systems Example 2 - Unsteady Nonlinear Navier-Stokes</h1>
// \author John W. Peterson
// \date 2004
//
// This example shows how a simple, unsteady, nonlinear system of equations
// can be solved in parallel.  The system of equations are the familiar
// Navier-Stokes equations for low-speed incompressible fluid flow.  This
// example introduces the concept of the inner nonlinear loop for each
// timestep, and requires a good deal of linear algebra number-crunching
// at each step.  If you have a ExodusII viewer such as ParaView installed,
// the script movie.sh in this directory will also take appropriate screen
// shots of each of the solution files in the time sequence.  These rgb files
// can then be animated with the "animate" utility of ImageMagick if it is
// installed on your system.  On a PIII 1GHz machine in debug mode, this
// example takes a little over a minute to run.  If you would like to see
// a more detailed time history, or compute more timesteps, that is certainly
// possible by changing the n_timesteps and dt variables below.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <sstream>
#include <math.h>

// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
#include "libmesh/perf_log.h"
#include "libmesh/boundary_info.h"
#include "libmesh/utility.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/zero_function.h"
#include "libmesh/const_function.h"
#include "libmesh/parsed_function.h"

// For systems of equations the DenseSubMatrix
// and DenseSubVector provide convenient ways for
// assembling the element matrix and vector on a
// component-by-component basis.
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"

// The definition of a geometric element
#include "libmesh/elem.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Function prototype.  This function will assemble the system
// matrix and right-hand-side.
void assemble_stokes (EquationSystems & es,
                      const std::string & system_name);

// Functions which set Dirichlet BCs corresponding to different problems.
void set_lid_driven_bcs(TransientLinearImplicitSystem & system);
void set_stagnation_bcs(TransientLinearImplicitSystem & system);
void set_poiseuille_bcs(TransientLinearImplicitSystem & system);

// The main program.
int main (int argc, char** argv)
{
  LibMeshInit init (argc, argv);

  Mesh mesh(init.comm());

  MeshTools::Generation::build_cube (mesh,
                                       10, 10, 10,
                                       0., 1.,
                                       0., 1.,
                                       0., 1.,
                                       HEX27);

  mesh.print_info();

  EquationSystems equation_systems (mesh);

  TransientLinearImplicitSystem & system =
    equation_systems.add_system<TransientLinearImplicitSystem> ("Navier-Stokes");

  system.add_variable ("velx", SECOND);
  system.add_variable ("vely", SECOND);
  system.add_variable ("velz", SECOND);
  system.add_variable ("p", FIRST);

  system.attach_assemble_function (assemble_stokes);

  // set_lid_driven_bcs(system);
  // set_stagnation_bcs(system);
  set_poiseuille_bcs(system);

  equation_systems.init ();

  equation_systems.print_info();

  PerfLog perf_log("stokes3d libmesh");

  TransientLinearImplicitSystem & navier_stokes_system =
    equation_systems.get_system<TransientLinearImplicitSystem>("Navier-Stokes");

  const Real dt = 0.5;
  navier_stokes_system.time     = 0.0;
  const unsigned int n_timesteps = 20;

  const unsigned int n_nonlinear_steps = 1;
  const Real nonlinear_tolerance       = 1.e-5;

  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
  equation_systems.parameters.set<Real>("linear solver tolerance") = 1.e-12;

  equation_systems.parameters.set<Real> ("dt")   = dt;
  equation_systems.parameters.set<Real> ("nu") = .007;

  UniquePtr<NumericVector<Number> >
    last_nonlinear_soln (navier_stokes_system.solution->clone());

  ExodusII_IO exo_io(mesh);

  for (unsigned int t_step=1; t_step<=n_timesteps; ++t_step)
    {
      navier_stokes_system.time += dt;

      libMesh::out << "\n\n*** Solving time step "
                   << t_step
                   << ", time = "
                   << navier_stokes_system.time
                   << " ***"
                   << std::endl;

      *navier_stokes_system.old_local_solution = *navier_stokes_system.current_local_solution;

      const Real initial_linear_solver_tol = 1.e-6;
      equation_systems.parameters.set<Real> ("linear solver tolerance") = initial_linear_solver_tol;

      bool converged = false;

      for (unsigned int l=0; l<n_nonlinear_steps; ++l)
        {
          last_nonlinear_soln->zero();
          last_nonlinear_soln->add(*navier_stokes_system.solution);

          perf_log.push("linear solve");
          equation_systems.get_system("Navier-Stokes").solve();
          perf_log.pop("linear solve");

          last_nonlinear_soln->add (-1., *navier_stokes_system.solution);
          last_nonlinear_soln->close();
          const Real norm_delta = last_nonlinear_soln->l2_norm();

          const unsigned int n_linear_iterations = navier_stokes_system.n_linear_iterations();

          const Real final_linear_residual = navier_stokes_system.final_linear_residual();

          if (n_linear_iterations == 0 &&
              (navier_stokes_system.final_linear_residual() >= nonlinear_tolerance || l==0))
            {
              Real old_linear_solver_tolerance = equation_systems.parameters.get<Real> ("linear solver tolerance");
              equation_systems.parameters.set<Real> ("linear solver tolerance") = 1.e-3 * old_linear_solver_tolerance;
              continue;
            }

          libMesh::out << "Linear solver converged at step: "
                       << n_linear_iterations
                       << ", final residual: "
                       << final_linear_residual
                       << "  Nonlinear convergence: ||u - u_old|| = "
                       << norm_delta
                       << std::endl;

          if ((norm_delta < nonlinear_tolerance) &&
              (navier_stokes_system.final_linear_residual() < nonlinear_tolerance))
            {
              libMesh::out << " Nonlinear solver converged at step "
                           << l
                           << std::endl;
              converged = true;
              break;
            }

          Real new_linear_solver_tolerance = std::min(Utility::pow<2>(final_linear_residual), initial_linear_solver_tol);
          equation_systems.parameters.set<Real> ("linear solver tolerance") = new_linear_solver_tolerance;
        } // end nonlinear loop

      // Don't keep going if we failed to converge.
//      if (!converged)
//        libmesh_error_msg("Error: Newton iterations failed to converge!");

#ifdef LIBMESH_HAVE_EXODUS_API
      // Write out every nth timestep to file.
      const unsigned int write_interval = 1;

      if ((t_step+1)%write_interval == 0)
        {
          exo_io.write_timestep("out_libmesh.e",
                                equation_systems,
                                t_step+1, // we're off by one since we wrote the IC and the Exodus numbering is 1-based.
                                navier_stokes_system.time);
        }
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
    } // end timestep loop.

  // All done.
  return 0;
}






// The matrix assembly function to be called at each time step to
// prepare for the linear solve.
void assemble_stokes (EquationSystems & es,
                      const std::string & libmesh_dbg_var(system_name))
{
  const MeshBase & mesh = es.get_mesh();
  const unsigned int dim = mesh.mesh_dimension();

  TransientLinearImplicitSystem & navier_stokes_system =
    es.get_system<TransientLinearImplicitSystem> ("Navier-Stokes");

  const unsigned int u_var = navier_stokes_system.variable_number ("velx");
  const unsigned int v_var = navier_stokes_system.variable_number ("vely");
  const unsigned int w_var = navier_stokes_system.variable_number ("velz");
  const unsigned int p_var = navier_stokes_system.variable_number ("p");

  FEType fe_vel_type = navier_stokes_system.variable_type(u_var);
  FEType fe_pres_type = navier_stokes_system.variable_type(p_var);

  UniquePtr<FEBase> fe_vel  (FEBase::build(dim, fe_vel_type));
  UniquePtr<FEBase> fe_pres (FEBase::build(dim, fe_pres_type));

  QGauss qrule (dim, fe_vel_type.default_quadrature_order());
  fe_vel->attach_quadrature_rule (&qrule);
  fe_pres->attach_quadrature_rule (&qrule);

  const std::vector<Real> & JxW = fe_vel->get_JxW();
  const std::vector<std::vector<Real> > & phi = fe_vel->get_phi();
  const std::vector<std::vector<RealGradient> > & dphi = fe_vel->get_dphi();
  const std::vector<std::vector<Real> > & psi = fe_pres->get_phi();
  // const std::vector<std::vector<RealGradient> > & dpsi = fe_pres->get_dphi();

  const DofMap & dof_map = navier_stokes_system.get_dof_map();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
    Kuu(Ke), Kuv(Ke), Kuw(Ke), Kup(Ke),
    Kvu(Ke), Kvv(Ke), Kvw(Ke), Kvp(Ke),
    Kwu(Ke), Kwv(Ke), Kww(Ke), Kwp(Ke),
    Kpu(Ke), Kpv(Ke), Kpw(Ke), Kpp(Ke);

  DenseSubVector<Number>
    Fu(Fe),
    Fv(Fe),
    Fw(Fe),
    Fp(Fe);

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_w;
  std::vector<dof_id_type> dof_indices_p;

  const Real dt    = es.parameters.get<Real>("dt");
  const Real theta = 1.;
  const Real nu = es.parameters.get<Real>("nu");
  const bool pin_pressure = es.parameters.get<bool>("pin_pressure");

  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el; ++el)
    {
      const Elem * elem = *el;

      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_u, u_var);
      dof_map.dof_indices (elem, dof_indices_v, v_var);
      dof_map.dof_indices (elem, dof_indices_w, v_var);
      dof_map.dof_indices (elem, dof_indices_p, p_var);

      const unsigned int n_dofs   = dof_indices.size();
      const unsigned int n_u_dofs = dof_indices_u.size();
      const unsigned int n_v_dofs = dof_indices_v.size();
      const unsigned int n_w_dofs = dof_indices_w.size();
      const unsigned int n_p_dofs = dof_indices_p.size();

      fe_vel->reinit  (elem);
      fe_pres->reinit (elem);

      Ke.resize (n_dofs, n_dofs);
      Fe.resize (n_dofs);

      Kuu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
      Kuv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
      Kuw.reposition (u_var*n_u_dofs, w_var*n_u_dofs, n_u_dofs, n_w_dofs);
      Kup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);

      Kvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
      Kvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
      Kvw.reposition (v_var*n_v_dofs, w_var*n_v_dofs, n_v_dofs, n_w_dofs);
      Kvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);

      Kwu.reposition (w_var*n_w_dofs, u_var*n_w_dofs, n_w_dofs, n_u_dofs);
      Kwv.reposition (w_var*n_w_dofs, v_var*n_w_dofs, n_w_dofs, n_v_dofs);
      Kww.reposition (w_var*n_w_dofs, w_var*n_w_dofs, n_w_dofs, n_w_dofs);
      Kwp.reposition (w_var*n_w_dofs, p_var*n_w_dofs, n_w_dofs, n_p_dofs);

      Kpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
      Kpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
      Kpw.reposition (p_var*n_u_dofs, w_var*n_u_dofs, n_p_dofs, n_w_dofs);
      Kpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);

      Fu.reposition (u_var*n_u_dofs, n_u_dofs);
      Fv.reposition (v_var*n_u_dofs, n_v_dofs);
      Fw.reposition (w_var*n_u_dofs, n_w_dofs);
      Fp.reposition (p_var*n_u_dofs, n_p_dofs);

      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          Number u = 0., u_old = 0.;
          Number v = 0., v_old = 0.;
          Number w = 0., w_old = 0.;
          Number p_old = 0.;
          Gradient grad_u, grad_u_old;
          Gradient grad_v, grad_v_old;
          Gradient grad_w, grad_w_old;

          for (unsigned int l=0; l<n_u_dofs; l++)
            {
              u_old += phi[l][qp]*navier_stokes_system.old_solution (dof_indices_u[l]);
              v_old += phi[l][qp]*navier_stokes_system.old_solution (dof_indices_v[l]);
              w_old += phi[l][qp]*navier_stokes_system.old_solution (dof_indices_w[l]);
              grad_u_old.add_scaled (dphi[l][qp],navier_stokes_system.old_solution (dof_indices_u[l]));
              grad_v_old.add_scaled (dphi[l][qp],navier_stokes_system.old_solution (dof_indices_v[l]));
              grad_w_old.add_scaled (dphi[l][qp],navier_stokes_system.old_solution (dof_indices_w[l]));

              // From the previous Newton iterate:
              u += phi[l][qp]*navier_stokes_system.current_solution (dof_indices_u[l]);
              v += phi[l][qp]*navier_stokes_system.current_solution (dof_indices_v[l]);
              v += phi[l][qp]*navier_stokes_system.current_solution (dof_indices_w[l]);
              grad_u.add_scaled (dphi[l][qp],navier_stokes_system.current_solution (dof_indices_u[l]));
              grad_v.add_scaled (dphi[l][qp],navier_stokes_system.current_solution (dof_indices_v[l]));
              grad_w.add_scaled (dphi[l][qp],navier_stokes_system.current_solution (dof_indices_w[l]));
            }

          // Compute the old pressure value at this quadrature point.
          for (unsigned int l=0; l<n_p_dofs; l++)
            p_old += psi[l][qp]*navier_stokes_system.old_solution (dof_indices_p[l]);

          const NumberVectorValue U_old (u_old, v_old, w_old);
          const NumberVectorValue U     (u,     v,     w);
          const Number u_x = grad_u(0);
          const Number u_y = grad_u(1);
          const Number u_z = grad_u(2);
          const Number v_x = grad_v(0);
          const Number v_y = grad_v(1);
          const Number v_z = grad_v(2);
          const Number w_x = grad_w(0);
          const Number w_y = grad_w(1);
          const Number w_z = grad_w(2);

          for (unsigned int i=0; i<n_u_dofs; i++)
            {
              Fu(i) += JxW[qp]*(u_old*phi[i][qp] /*-
                                (1.-theta)*dt*(U_old*grad_u_old)*phi[i][qp] +
                                (1.-theta)*dt*p_old*dphi[i][qp](0)  -
                                (1.-theta)*dt*nu*(grad_u_old*dphi[i][qp])*/ /*+
                                theta*dt*(U*grad_u)*phi[i][qp]*/);

              Fv(i) += JxW[qp]*(v_old*phi[i][qp] /*-
                                (1.-theta)*dt*(U_old*grad_v_old)*phi[i][qp] +
                                (1.-theta)*dt*p_old*dphi[i][qp](1) -
                                (1.-theta)*dt*nu*(grad_v_old*dphi[i][qp])*/ /*+
                                theta*dt*(U*grad_v)*phi[i][qp]*/);

              Fw(i) += JxW[qp]*(w_old*phi[i][qp] /*-
                                (1.-theta)*dt*(U_old*grad_w_old)*phi[i][qp] +
                                (1.-theta)*dt*p_old*dphi[i][qp](2) -
                                (1.-theta)*dt*nu*(grad_w_old*dphi[i][qp])*/ /*+
                                theta*dt*(U*grad_w)*phi[i][qp]*/);

              for (unsigned int j=0; j<n_u_dofs; j++)
                {
                  Kuu(i,j) += JxW[qp]*(phi[i][qp]*phi[j][qp] +
                                       theta*dt*nu*(dphi[i][qp]*dphi[j][qp]) /*+
                                       theta*dt*(U*dphi[j][qp])*phi[i][qp] +
                                       theta*dt*u_x*phi[i][qp]*phi[j][qp]*/);

//                  Kuv(i,j) += JxW[qp]*theta*dt*u_y*phi[i][qp]*phi[j][qp];
//                  Kuw(i,j) += JxW[qp]*theta*dt*u_z*phi[i][qp]*phi[j][qp];

                  Kvv(i,j) += JxW[qp]*(phi[i][qp]*phi[j][qp] /*+
                                       theta*dt*nu*(dphi[i][qp]*dphi[j][qp]) +
                                       theta*dt*(U*dphi[j][qp])*phi[i][qp] +
                                       theta*dt*v_y*phi[i][qp]*phi[j][qp]*/);

//                  Kvu(i,j) += JxW[qp]*theta*dt*v_x*phi[i][qp]*phi[j][qp];
//                  Kvw(i,j) += JxW[qp]*theta*dt*v_z*phi[i][qp]*phi[j][qp];

                  Kww(i,j) += JxW[qp]*(phi[i][qp]*phi[j][qp] /*+
                                       theta*dt*nu*(dphi[i][qp]*dphi[j][qp]) +
                                       theta*dt*(U*dphi[j][qp])*phi[i][qp] +
                                       theta*dt*w_z*phi[i][qp]*phi[j][qp]*/);

//                  Kwu(i,j) += JxW[qp]*theta*dt*w_x*phi[i][qp]*phi[j][qp];
//                  Kwv(i,j) += JxW[qp]*theta*dt*w_y*phi[i][qp]*phi[j][qp];
                }

              for (unsigned int j=0; j<n_p_dofs; j++)
                {
                  Kup(i,j) += JxW[qp]*(-theta*dt*psi[j][qp]*dphi[i][qp](0));
                  Kvp(i,j) += JxW[qp]*(-theta*dt*psi[j][qp]*dphi[i][qp](1));
                  Kwp(i,j) += JxW[qp]*(-theta*dt*psi[j][qp]*dphi[i][qp](2));
                }
            }

          for (unsigned int i=0; i<n_p_dofs; i++)
            for (unsigned int j=0; j<n_u_dofs; j++)
              {
                Kpu(i,j) += -JxW[qp]*dt*psi[i][qp]*dphi[j][qp](0);
                Kpv(i,j) += -JxW[qp]*dt*psi[i][qp]*dphi[j][qp](1);
                Kpw(i,j) += -JxW[qp]*dt*psi[i][qp]*dphi[j][qp](2);
              }
        } // end of the quadrature point qp-loop

//      if (pin_pressure)
//        {
//          const Real penalty = 1.e10;
//          const unsigned int pressure_node = 0;
//          const Real p_value               = 0.0;
//          for (unsigned int c=0; c<elem->n_nodes(); c++)
//            if (elem->node_id(c) == pressure_node)
//              {
//                Kpp(c,c) += penalty;
//                Fp(c)    += penalty*p_value;
//              }
//        }

      dof_map.heterogenously_constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      navier_stokes_system.matrix->add_matrix (Ke, dof_indices);
      navier_stokes_system.rhs->add_vector    (Fe, dof_indices);
    } // end of element loop
}



// void set_lid_driven_bcs(TransientLinearImplicitSystem & system)
// {
//   unsigned short int
//     u_var = system.variable_number("vel_x"),
//     v_var = system.variable_number("vel_y");
//
//   // This problem *does* require a pressure pin, there are Dirichlet
//   // boundary conditions for u and v on the entire boundary.
//   system.get_equation_systems().parameters.set<bool>("pin_pressure") = true;
//
//   // Get a convenient reference to the System's DofMap
//   DofMap & dof_map = system.get_dof_map();
//
//   {
//     // u=v=0 on bottom, left, right
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(0);
//     boundary_ids.insert(1);
//     boundary_ids.insert(3);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(u_var);
//     variables.push_back(v_var);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ZeroFunction<Number>()));
//   }
//   {
//     // u=1 on top
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(2);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(u_var);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ConstFunction<Number>(1.)));
//   }
//   {
//     // v=0 on top
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(2);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(v_var);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ZeroFunction<Number>()));
//   }
// }



// void set_stagnation_bcs(TransientLinearImplicitSystem & system)
// {
//   unsigned short int
//     u_var = system.variable_number("vel_x"),
//     v_var = system.variable_number("vel_y");
//
//   // This problem does not require a pressure pin, the Neumann outlet
//   // BCs are sufficient to set the value of the pressure.
//   system.get_equation_systems().parameters.set<bool>("pin_pressure") = false;
//
//   // Get a convenient reference to the System's DofMap
//   DofMap & dof_map = system.get_dof_map();
//
//   {
//     // u=v=0 on bottom
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(0);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(u_var);
//     variables.push_back(v_var);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ZeroFunction<Number>()));
//   }
//   {
//     // u=0 on left (symmetry)
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(3);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(u_var);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ZeroFunction<Number>()));
//   }
//   {
//     // u = k*x on top
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(2);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(u_var);
//
//     // Set up ParsedFunction parameters
//     std::vector<std::string> additional_vars;
//     additional_vars.push_back("k");
//     std::vector<Number> initial_vals;
//     initial_vals.push_back(1.);
//
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ParsedFunction<Number>("k*x",
//                                                                             &additional_vars,
//                                                                             &initial_vals)));
//   }
//   {
//     // v = -k*y on top
//     std::set<boundary_id_type> boundary_ids;
//     boundary_ids.insert(2);
//
//     std::vector<unsigned int> variables;
//     variables.push_back(v_var);
//
//     // Set up ParsedFunction parameters
//     std::vector<std::string> additional_vars;
//     additional_vars.push_back("k");
//     std::vector<Number> initial_vals;
//     initial_vals.push_back(1.);
//
//     // Note: we have to specify LOCAL_VARIABLE_ORDER here, since we're
//     // using a ParsedFunction to set the value of v_var, which is
//     // actually the second variable in the system.
//     dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
//                                                      variables,
//                                                      ParsedFunction<Number>("-k*y",
//                                                                             &additional_vars,
//                                                                             &initial_vals),
//                                                      LOCAL_VARIABLE_ORDER));
//   }
// }



void set_poiseuille_bcs(TransientLinearImplicitSystem & system)
{
  unsigned short int
    u_var = system.variable_number("velx"),
    v_var = system.variable_number("vely"),
    w_var = system.variable_number("velz");

  system.get_equation_systems().parameters.set<bool>("pin_pressure") = false;

  DofMap & dof_map = system.get_dof_map();

  {
    // u=v=0 on top, bottom
    std::set<boundary_id_type> boundary_ids = {0, 1, 3, 5};

    std::vector<uint> variables = {u_var, v_var, w_var};

    dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
                                                     variables,
                                                     ZeroFunction<Number>()));
  }
  {
    // u=quadratic on left
    std::set<boundary_id_type> boundary_ids = {4};

    std::vector<unsigned int> variables = {u_var};

    dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
                                                     variables,
                                                     ParsedFunction<Number>("16*y*(1-y)*z*(1-z)")));
  }
  {
    // v=0 on left
    std::set<boundary_id_type> boundary_ids = {4};

    std::vector<unsigned int> variables = {v_var, w_var};

    dof_map.add_dirichlet_boundary(DirichletBoundary(boundary_ids,
                                                     variables,
                                                     ZeroFunction<Number>()));
  }
}
