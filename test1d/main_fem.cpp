#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/fem_system.h>
#include <libmesh/quadrature.h>
#include <libmesh/euler_solver.h>
#include <libmesh/steady_solver.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/zero_function.h>
#include <libmesh/const_function.h>
#include <libmesh/dof_map.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/diff_solver.h>
#include <libmesh/sparse_matrix.h>

#include <iostream>
#include <iomanip>

using namespace libMesh;

Real exact_solution(Point const p)
{
    return std::sin(0.5*M_PI*p(0));
}

class Diff: public FEMSystem
{
public:

    Diff(EquationSystems& es, std::string const& name, uint const number):
        FEMSystem(es, name, number)
    {}

    virtual void init_data()
    {
        _u_var = this->add_variable("u", Order::FIRST, FEFamily::LAGRANGE);

        this->time_evolving(_u_var);

        ZeroFunction<Number> zero;
        ConstFunction<Number> one(1.0);

        this->get_dof_map().add_dirichlet_boundary(DirichletBoundary({0}, {0}, &zero));
//        this->get_dof_map().add_dirichlet_boundary(DirichletBoundary({1}, {0}, &one));

        FEMSystem::init_data();
    }

    virtual void init_context(DiffContext& context)
    {
        FEMContext& c = cast_ref<FEMContext&>(context);

        FEBase* u_elem_fe;

        c.get_element_fe(_u_var, u_elem_fe);

        u_elem_fe->get_JxW();
        u_elem_fe->get_phi();
        u_elem_fe->get_dphi();
        u_elem_fe->get_xyz();

        FEMSystem::init_context(context);
    }

    virtual bool element_time_derivative(bool request_jacobian, DiffContext& context)
    {
        FEMContext& c = cast_ref<FEMContext&>(context);

        FEBase* u_elem_fe;

        c.get_element_fe(_u_var, u_elem_fe);

        std::vector<Real> const& JxW = u_elem_fe->get_JxW();
        std::vector<std::vector<Real>> const& v = u_elem_fe->get_phi();
        std::vector<std::vector<RealGradient>> const& dv = u_elem_fe->get_dphi();
        std::vector<Point> const& qpoint = u_elem_fe->get_xyz();

        uint const n_dofs = c.get_dof_indices(_u_var).size();

        DenseSubMatrix<Number>& Ke = c.get_elem_jacobian(_u_var, _u_var);
        DenseSubVector<Number>& Fe = c.get_elem_residual(_u_var);

        uint n_qp = c.get_element_qrule().n_points();

        for(uint qp = 0; qp < n_qp; qp++)
        {
            Gradient du = c.interior_gradient(_u_var, qp);

            Real f = this->forcing(qpoint[qp]);

            for(uint i = 0; i < n_dofs; i++)
            {
                Fe(i) += JxW[qp] * (du*dv[i][qp] - f*v[i][qp]);

                // if(request_jacobian && c.elem_solution_derivative)
                if(request_jacobian)
                {
                    for(uint j = 0; j < n_dofs; j++)
                    {
                        Ke(i,j) += JxW[qp] * (dv[j][qp] * dv[i][qp]);
                    }
                }
            }
        }

        return request_jacobian;
    }

//    virtual bool side_time_derivative (bool request_jacobian, DiffContext& /*context*/)
//    {
//        return request_jacobian;
//    }

//    virtual bool mass_residual (bool request_jacobian, DiffContext& context)
//    {
//        FEMContext& c = cast_ref<FEMContext&>(context);

//        FEBase* u_elem_fe;

//        c.get_element_fe(_u_var, u_elem_fe);

//        std::vector<Real> const& JxW = u_elem_fe->get_JxW();
//        std::vector<std::vector<Real>> const& v = u_elem_fe->get_phi();
//        uint const n_dofs = c.get_dof_indices(_u_var).size();

//        DenseSubMatrix<Number>& Ke = c.get_elem_jacobian(_u_var, _u_var);
//        DenseSubVector<Number>& Fe = c.get_elem_residual(_u_var);

//        uint n_qp = c.get_element_qrule().n_points();

//        for(uint qp = 0; qp < n_qp; qp++)
//        {
//            Real u_old = c.interior_value( _u_var, qp );

//            for(uint i = 0; i < n_dofs; i++)
//            {
//                Fe(i) += JxW[qp] * u_old * v[i][qp];

////                if(request_jacobian && c.elem_solution_derivative)
//                if(request_jacobian)
//                {
//                    for(uint j = 0; j < n_dofs; j++)
//                    {
//                        Ke(i,j) += JxW[qp] * v[j][qp] * v[i][qp];
//                    }
//                }
//            }
//        }
//        return request_jacobian;
//    }

    Real forcing(Point const& p)
    {
        return 0.25*M_PI*M_PI*std::sin(0.5*M_PI*p(0));
    }

//    virtual void postprocess()
//    {}

private:
    uint _u_var;

};

int main(int argc, char* argv[])
{
    LibMeshInit init(argc, argv);

    Mesh mesh(init.comm());

    MeshTools::Generation::build_line(mesh, 20, 0., 1., EDGE2);
    //MeshTools::Generation::build_square(mesh, 4, 10, 0., 1., 0., 1., QUAD4);

    mesh.print_info();

    mesh.boundary_info->print_info();

    EquationSystems es(mesh);

    Diff& system = es.add_system<Diff>("Diff");

//     system.time_solver = UniquePtr<TimeSolver>(new EulerSolver(system));
//     system.deltat= 1.0;
    system.time_solver = UniquePtr<TimeSolver>(new SteadySolver(system));

    es.init();

    DiffSolver& solver = *(system.time_solver->diff_solver().get());
    solver.quiet = false;
    solver.verbose = !solver.quiet;
    solver.max_nonlinear_iterations = 15;
    solver.relative_step_tolerance = 1.e-3;
    solver.relative_residual_tolerance = 0.0;
    solver.absolute_residual_tolerance = 0.0;
    solver.max_linear_iterations = 50000;
    solver.initial_linear_tolerance = 1.e-3;

    es.print_info();

    system.solve();

    system.matrix->print();
    system.rhs->print();
    system.solution->print();

    {
        std::ostringstream file_name;

        file_name << "out_"
                  << std::setw(3)
                  << std::setfill('0')
                  << std::right
                  << 0
                  << ".e";

        ExodusII_IO(mesh).write_timestep(file_name.str(), es, 1, 0.0/*system.time*/);
    }

    // const uint n_timesteps = 1;
    //
    // for (uint t_step=0; t_step != n_timesteps; ++t_step)
    // {
    //     std::cout << "\n\nSolving time step " << t_step << ", time = "
    //               << system.time << std::endl;
    //
    //     system.solve();
    //
    //     system.solution->print();
    //
    //     system.time_solver->advance_timestep();
    //
    //     {
    //       std::ostringstream file_name;
    //
    //       // We write the file in the ExodusII format.
    //       file_name << "out_"
    //                 << std::setw(3)
    //                 << std::setfill('0')
    //                 << std::right
    //                 << t_step+1;
    //
    //       ExodusII_IO(mesh).write_timestep(file_name.str(), es, 1, system.time);
    //     }
    // }

    return 0;
}
