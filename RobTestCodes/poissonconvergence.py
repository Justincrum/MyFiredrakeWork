#Test codes courtesty of Rob.
# Test convergence for Poisson problem with Nitsche BC
from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
import numpy
import os
​
if COMM_WORLD.rank == 0:
    if not os.path.exists("data"):
        os.makedirs("data")
    elif not os.path.isdir("data"):
        raise RuntimeError("Cannot create output directory, file of given name exists")
COMM_WORLD.barrier()
​
els = ["Lagrange", "Lagrange", "Lagrange", "Hermite", "Bell", "Argyris"]
degs = [3, 4, 5, 3, 5, 5]
​
PETSc.Log().begin()
​
​
def get_time(event, comm=COMM_WORLD):
    return comm.allreduce(PETSc.Log.Event(event).getPerfInfo()["time"], op=MPI.SUM) / comm.size
​
​
N_base = 8
mesh = UnitSquareMesh(N_base, N_base)
mh = MeshHierarchy(mesh, 4)
# Let's perturb the original mesh
V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
eps = Constant(3 / 2**(N_base-1))
​
x, y = SpatialCoordinate(mesh)
new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                         y - eps*sin(2*pi*x)*sin(2*pi*y)]))
​
# And propagate to refined meshes
coords = [new]
for mesh in mh[1:]:
    fine = Function(mesh.coordinates.function_space())
    prolong(new, fine)
    coords.append(fine)
    new = fine
​
for mesh, coord in zip(mh, coords):
    mesh.coordinates.assign(coord)
​
for el, deg in zip(els, degs):
    PETSc.Sys.Print("{}-{}".format(el, deg))
    results = []
    element = FiniteElement(el, triangle, deg)
    for i, msh in enumerate(mh):
        N = N_base * 2**i
        x, y = SpatialCoordinate(msh)
        uex = sin(pi*x)*sin(2*pi*y)
​
        f = -div(grad(uex))
​
        V = FunctionSpace(msh, element)
​
        n = FacetNormal(msh)
        h = CellSize(msh)
        beta = Constant(20)
​
        u = TrialFunction(V)
        v = TestFunction(V)
​
        a = (inner(grad(u), grad(v))*dx -
             inner(dot(grad(u), n), v)*ds -
             inner(u, dot(grad(v), n))*ds + inner(beta/h*u, v)*ds)
​
        L = inner(f, v)*dx
​
        uh = Function(V)
​
        params = {"snes_type": "newtonls",
                  "snes_linesearch_type": "basic",
                  "snes_max_it": 2,
                  "snes_lag_jacobian": -2,
                  "snes_lag_preconditioner": -2,
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "pastix",
                  "snes_rtol": 1e-16,
                  "snes_atol": 1e-25}
​
        if i == 0:
            # Warm up symbolics/disk cache
            solve(a == L, uh, solver_parameters=params)
​
        with PETSc.Log.Stage("{el}{deg}.N{N}".format(el=el, deg=deg, N=N)):
            solve(a == L, uh, solver_parameters=params, options_prefix="")
            snes = get_time("SNESSolve")
            ksp = get_time("KSPSolve")
            pcsetup = get_time("PCSetUp")
            pcapply = get_time("PCApply")
            jac = get_time("SNESJacobianEval")
            residual = get_time("SNESFunctionEval")
            sparsity = get_time("CreateSparsity")
​
        error = sqrt(assemble((uex-uh)**2*dx))
​
        results.append([N, error, snes, ksp, pcsetup, pcapply, jac, residual, sparsity])
​
    results = numpy.asarray(results)
    if mesh.comm.rank == 0:
        with open('data/poisson.{el}.{deg}.csv'.format(el=el, deg=deg), 'w') as f:
            numpy.savetxt(f, results, fmt=['%d'] + ['%e'] * 8, delimiter=',',
                          header='N,Error,SNESSolve,KSPSolve,PCSetUp,PCApply,SNESJacobianEval,SNESFunctionEval,CreateSparsity', comments='')