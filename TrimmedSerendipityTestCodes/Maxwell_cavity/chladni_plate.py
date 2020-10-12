# example usage
# python chladni.py -eps_nev 5 -eps_tol 1e-11 -eps_target 10 -st_type sinvert
from firedrake import *
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
N = 64
msh = RectangleMesh(N, N, 2, 2)
# Some finer meshes for visualisation
mh = MeshHierarchy(msh, 2)
for m in mh:
        m.coordinates.dat.data[:] -= 1
vizmesh = mh[-1]
deg = 5
V = FunctionSpace(msh, "Argyris", deg)
Pk = FunctionSpace(msh, "CG", deg)
u = TrialFunction(V)
v = TestFunction(V)
# Following Ritz (1909)
nu = Constant(0.225)
a = (inner(u.dx(0).dx(0), v.dx(0).dx(0)) +
    inner(u.dx(1).dx(1), v.dx(1).dx(1)) +
    nu*(inner(u.dx(1).dx(1), v.dx(0).dx(0)) +
    inner(u.dx(0).dx(0), v.dx(1).dx(1))) +
    2*(1 - nu)*inner(u.dx(0).dx(1), v.dx(0).dx(1)))*dx
mss = inner(u, v)*dx
# This prefix allows us to control factorisation package from options.
A = assemble(a, options_prefix="st_").M.handle
M = assemble(mss, options_prefix="st_").M.handle
E = SLEPc.EPS().create(comm=msh.comm)
E.setOperators(A, M)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
# Defaults  -eps_nev 5 -eps_tol 1e-11 -eps_target 10 -st_type sinvert
E.setTolerances(tol=1e-11)
E.setTarget(10)
E.setDimensions(nev=10)
E.st.setType(SLEPc.ST.Type.SINVERT)
E.setFromOptions()
E.solve()
Print = PETSc.Sys.Print
Print()
Print("******************************")
Print("*** SLEPc Solution Results ***")
Print("******************************")
Print()
its = E.getIterationNumber()
Print("Number of iterations of the method: %d" % its)
eps_type = E.getType()
Print("Solution method: %s" % eps_type)
nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %d" % nev)
tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
nconv = E.getConverged()
Print("Number of converged eigenpairs %d" % nconv)

if nconv > 0:
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    ks = []
    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        ks.append(k.real)
        inds = np.argsort(ks)

if nconv > 0:
# Create the results vectors
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    #
    Print()
    Print("        k          ||Ax-kx||/||kx||")
    Print("----------------- ------------------")
    for j, i in enumerate(inds):
        k = E.getEigenpair(i, vr, vi)
        error = E.computeError(i)
        if k.real < 1e-10:
            continue
        if k.imag != 0.0:
            Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
        else:
            Print(" %12f      %12g" % (k.real, error))
                                                                                                                            

