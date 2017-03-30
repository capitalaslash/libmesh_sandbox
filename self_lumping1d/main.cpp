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
#include <libmesh/const_function.h>

#include <libmesh/sparse_matrix.h>
#include <libmesh/numeric_vector.h>

#include <libmesh/error_vector.h>
#include <libmesh/kelly_error_estimator.h>

#include <libmesh/exodusII_io.h>

#include <iostream>
#include <iomanip>

using namespace libMesh;

Real force(Point const& p)
{
    return 0.25*M_PI*M_PI * std::cos(0.5*M_PI * p(0));
}

void assemble_poisson(EquationSystems& es, const std::string& system_name)
{
    const MeshBase& mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> (system_name);

    const DofMap& dof_map = system.get_dof_map();
    FEType fe_type = dof_map.variable_type(0);
    UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, SECOND);
    fe->attach_quadrature_rule (&qrule);

    UniquePtr<FEBase> fe_side (FEBase::build(dim, fe_type));
    QGauss qside (dim-1, SECOND);
    fe_side->attach_quadrature_rule (&qside);

    std::vector<Real> const& JxW = fe->get_JxW();

    std::vector<std::vector<Real>> const &              v = fe->get_phi();
    std::vector<std::vector<RealGradient>> const &     dv = fe->get_dphi();
    std::vector<Point> const &                     qpoint = fe->get_xyz();

//    std::vector<std::vector<Real>> const &      v_side = fe_side->get_phi();
//    std::vector<Real> const &                 JxW_side = fe_side->get_JxW();
//    std::vector<Point> const &             qside_point = fe_side->get_xyz();

    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;

    std::vector<dof_id_type> dof_indices;

    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for ( ; el != end_el ; ++el)
    {
        const Elem* elem = *el;

        dof_map.dof_indices (elem, dof_indices);

        fe->reinit (elem);

        uint const n_dofs = dof_indices.size();

        Ke.resize(n_dofs, n_dofs);
        Fe.resize(n_dofs);

        uint n_qp = qrule.n_points();

        for(uint qp = 0; qp < n_qp; qp++)
        {
            Real const f = force(qpoint[qp]);

            for(uint i = 0; i < n_dofs; i++)
            {
                Fe(i) += JxW[qp] * f * v[i][qp];

                for(uint j = 0; j < n_dofs; j++)
                {
                    Ke(i,j) += JxW[qp] * (dv[j][qp] * dv[i][qp]);
                }
            }
        }

//        for (uint side = 0; side<elem->n_sides(); side++)
//            if (elem->neighbor(side) == NULL)
//            {
//                fe_side->reinit(elem, side);
//
//                for (uint qp=0; qp<qside.n_points(); qp++)
//                {
//
//                    const Real xf = qside_point[qp](0);
//                    const Real yf = qside_point[qp](1);
//
//                    const Real penalty = 1.e10;
//
//                    const Real value = 0.0;
//
//                    if(yf < 1e-6)
//                    {
//                        for (uint i = 0; i < v_side.size(); i++)
//                        {
//                            Fe(i) += JxW_side[qp] * penalty * value * v_side[i][qp];
//                            for (uint j = 0; j < v_side.size(); j++)
//                                Ke(i,j) += JxW_side[qp] * penalty * v_side[i][qp] * v_side[j][qp];
////                            Fe(i) += JxW_side[qp] * 2.0 * v_side[i][qp];
//                        }
//
//                    }
//                }
//            }

        out << "Ke = \n" << Ke << std::endl;
        out << "Fe = \n" << Fe << std::endl;

        dof_map.heterogenously_constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

        out << "Ke = \n" << Ke << std::endl;
        out << "Fe = \n" << Fe << std::endl;

        system.matrix->add_matrix (Ke, dof_indices);
        system.rhs->add_vector    (Fe, dof_indices);
    }
    system.matrix->close();
    out << "system.matrix =\n" << *system.matrix << std::endl;
    out << "system.rhs =\n" << *system.rhs << std::endl;

}

int main(int argc, char* argv[])
{
    LibMeshInit init(argc, argv);

    Mesh mesh(init.comm());

    MeshTools::Generation::build_line(mesh, 2, 0., 1., EDGE2);

    mesh.print_info();

    mesh.boundary_info->print_info();

    EquationSystems es(mesh);

    LinearImplicitSystem& system = es.add_system<LinearImplicitSystem>("Diff");

    system.add_variable("u", FIRST, LAGRANGE);

    system.attach_assemble_function(assemble_poisson);

    ZeroFunction<Real> fzero;
    ConstFunction<Real> fc(1.0);

    DirichletBoundary dirichlet_bc_left({0}, {0}, &fc);
    DirichletBoundary dirichlet_bc_right({1}, {0}, &fzero);

    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc_left);
    system.get_dof_map().add_dirichlet_boundary(dirichlet_bc_right);

    es.init();

    es.print_info();

    system.solve();

    ErrorVector error;
    UniquePtr<ErrorEstimator> error_estimator(new KellyErrorEstimator);
    error_estimator->estimate_error(system, error);
    Real global_error = error.l2_norm();

    out << "l2 error = " << global_error << std::endl;

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
