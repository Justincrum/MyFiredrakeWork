#Code given by Rob.
from firedrake import *
import scipy
import matplotlib.pyplot as plt
import os
import numpy
import scipy.sparse

if not os.path.exists("pictures/sparsity"):
    os.makedirs("pictures/sparsity")
elif not os.path.isdir("pictures/sparsity"):
    raise RuntimeError("Cannot create output directory, file of given name exists")

N = 8
msh = UnitSquareMesh(N, N)

FONTSIZE = 18

stuff = []

for el, deg in zip(["Lagrange"]*3,
                   [3, 4, 5]):
    if el == "Lagrange":
        name = '{$P^%d$}' % deg
    else:
        name = el

    V = FunctionSpace(msh, el, deg)

    n = FacetNormal(msh)
    h = CellSize(msh)
    beta = Constant(20)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = (inner(grad(u), grad(v))*dx -
         inner(dot(grad(u), n), v)*ds -
         inner(u, dot(grad(v), n))*ds +
         inner(beta/h*u, v)*ds)

    A = assemble(a).M.handle
    nrow = A.getSize()[0]
    ai, aj, av = A.getValuesCSR()
    Asp = scipy.sparse.csr_matrix((av, aj, ai))

    plt.spy(Asp, markersize=0.5, aspect="equal", precision="present",
            color="black")
    ax = plt.gca()
    ax.set_xticks([0, nrow])
    ax.set_yticks([0, nrow])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE)
    plt.savefig("pictures/sparsity/poisson-{el}-{deg}.png".format(el=el, deg=deg),
                format="png", bbox_inches="tight", pad_inches=0,
                dpi=200, background="transparent", fontsize=FONTSIZE)

    # now print out some data about conditioning and nonzeros
    nnz = Asp.nnz
    nrows = Asp.shape[0]
    kappa = numpy.linalg.cond(Asp.todense())
    stuff.append([name, nrows, nnz, nnz/float(nrows), kappa])

    if el == "Lagrange":
        sparams =  {"mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200", "ksp_type": "gmres", "ksp_monitor_true_residual": None, "ksp_converged_reason": None}

        problem = LinearVariationalProblem(a, 0, Function(V))
        solver = LinearVariationalSolver(problem, solver_parameters=sparams)
        solver.snes.setUp()
        solver.snes.ksp.setUp()
        S = solver.snes.ksp.pc.getPythonContext().S.petscmat
        nrow = S.getSize()[0]
        ai, aj, av = S.getValuesCSR()
        Asp = scipy.sparse.csr_matrix((av, aj, ai))
        plt.spy(Asp, markersize=0.5, aspect="equal", precision="present",
                color="black")
        ax = plt.gca()
        ax.set_xticks([0, nrow])
        ax.set_yticks([0, nrow])
        for tick in ax.get_xticklabels():
            tick.set_fontsize(FONTSIZE)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(FONTSIZE)
        plt.savefig("pictures/sparsity/poisson-condensed-{el}-{deg}.png".format(el=el, deg=deg),
                    format="png", bbox_inches="tight", pad_inches=0,
                    dpi=200, background="transparent", fontsize=FONTSIZE)

        nnz = Asp.nnz
        nrows = Asp.shape[0]
        kappa = numpy.linalg.cond(Asp.todense())

        name = '{$P_c^%d$}' % deg
        stuff.append([name, nrows, nnz, nnz/float(nrows), kappa])

with open('data/poisson-spy-cond.csv', 'w') as f:
    numpy.savetxt(f, stuff, fmt=['%s', '%s', '%s', '%s', '%s'], delimiter=',',
                  header='Name, Nrows, NNZ, NNZperRow, kappa', comments='')