#Modified test code to work on Mixed Poisson problem.  Want to test Sminus Div elements in 2 and 3 dimensions.

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
for n in range(6, 7):
    ErrorValues = []
    GridSize = []
    RateValues = []
    for j in range(3, 8):
        PolyDegree = n
        Cells = 2**j
        mesh = UnitSquareMesh(Cells, Cells, quadrilateral=True)

        V = FunctionSpace(mesh, "Lagrange", PolyDegree)
        u = TrialFunction(V)
        v = TestFunction(V)
        DOFs = V.dim()

        x, y = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)
        f = -div(grad(uex))

        a = inner(grad(u), grad(v))*dx
        bc = DirichletBC(V.sub(0), 0, "on_boundary")
        L = inner(f, v)*dx
        uh = Function(V)

        params = {"mat_type": "aij", "snes_type": "newtonls", "snes_max_it": "2", "snes_convergence_test": "skip", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        solve(a == L, uh, bcs=bc, solver_parameters=params)

        ErrVal = norms.errornorm(uex, uh)
        ErrorValues.append(ErrVal)
        GridSize.append(1 / 2**j)
        Info = [[PolyDegree], [Cells], [DOFs], [ErrVal]]
        file=open("2d-results-primal-poisson-Lagrange.csv", 'a', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(Info)
    for j in range(0, len(ErrorValues)-1):
        top = np.log(ErrorValues[j]/ErrorValues[j+1])
        bottom = np.log(GridSize[j] / GridSize[j+1])
        rate = top / bottom
        RateValues.append(rate)
    print(ErrorValues)
    print(RateValues)
#print(ErrVal, SigErrVal)
