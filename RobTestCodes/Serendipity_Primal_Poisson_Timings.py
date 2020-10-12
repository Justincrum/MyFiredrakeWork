#Code given by Rob.
# Test run-times for Poisson with P^k and static condensation
from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
import numpy
import os

if COMM_WORLD.rank == 0:
    if not os.path.exists("data"):
        os.makedirs("data")
    elif not os.path.isdir("data"):
        raise RuntimeError("Cannot create output directory, file of given name exists")
COMM_WORLD.barrier()

#one_form_els = ["SminusDiv"]*3
#two_form_els = ["DPC"]*3
degs = range(2, 7)

PETSc.Log().begin()


def get_time(event, comm=COMM_WORLD):
    return comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"], op=MPI.SUM) / comm.size


N_base = 8
mesh = UnitSquareMesh(N_base, N_base, quadrilateral=True)
mh = MeshHierarchy(mesh, 4)
# Let's perturb the original mesh
V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
eps = Constant(3 / 2**(N_base-1))

x, y = SpatialCoordinate(mesh)
new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                         y - eps*sin(2*pi*x)*sin(2*pi*y)]))

# And propagate to refined meshes
coords = [new]
for mesh in mh[1:]:
    fine = Function(mesh.coordinates.function_space())
    prolong(new, fine)
    coords.append(fine)
    new = fine

for mesh, coord in zip(mh, coords):
    mesh.coordinates.assign(coord)

#for el_one, el_two, deg in list(zip(one_form_els, two_form_els, degs)):
for deg in degs:
    #PETSc.Sys.Print("{}-{}".format(deg))
    results = []
    #element = FiniteElement(el, triangle, deg)
    for i, msh in enumerate(mh):
        N = N_base * 2**i
        x, y = SpatialCoordinate(msh)
        uex = sin(pi*x)*sin(pi*y)
        sigmaex = grad(uex)
        f = -div(grad(uex))

        Sminus = FunctionSpace(msh, "SminusDiv", deg)
        DPC = FunctionSpace(msh, "DPC", deg-1)
        V = Sminus * DPC

        n = FacetNormal(msh)
        h = CellSize(msh)
        beta = Constant(20)

        sigma, u = TrialFunctions(V)
        tau, v = TestFunctions(V)

        a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
        L = -f * v * dx
        #L = inner(f, v)*dx

        w = Function(V)
        params = {"mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200", "ksp_type": "gmres", "ksp_monitor_true_residual": None, "ksp_converged_reason": None}


        if i == 0:
            # Warm up symbolics/disk cache
            solve(a == L, w, solver_parameters=params)
            w.assign(0)
        
        el = "Sminus*DPC"

        with PETSc.Log.Stage("{el}{deg}.N{N}".format(el=el, deg=deg, N=N)):
            solve(a == L, w, solver_parameters=params, options_prefix="")
            snes = get_time("SNESSolve")
            ksp = get_time("KSPSolve")
            pcsetup = get_time("PCSetUp")
            pcapply = get_time("PCApply")
            jac = get_time("SNESJacobianEval")
            residual = get_time("SNESFunctionEval")
            sparsity = get_time("CreateSparsity")

        sigma, u = w.split()
        error = sqrt(assemble((uex-u)**2*dx))

        results.append([N, error, snes, ksp, pcsetup, pcapply, jac, residual, sparsity])

    results = numpy.asarray(results)

    if mesh.comm.rank == 0:
        with open('data/poisson.condensed.{el}.{deg}.csv'.format(el=el, deg=deg), 'w') as f:
            numpy.savetxt(f, results, fmt=['%d'] + ['%e'] * 8, delimiter=',',
                          header='N,Error,SNESSolve,KSPSolve,PCSetUp,PCApply,SNESJacobianEval,SNESFunctionEval,CreateSparsity', comments='')
