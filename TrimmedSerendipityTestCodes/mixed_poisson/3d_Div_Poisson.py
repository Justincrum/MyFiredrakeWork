#Modified test code to work on Mixed Poisson problem.  Want to test Sminus Div elements in 2 and 3 dimensions.

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
for n in range(1, 4):
    for j in range(3, 7):
        PolyDegree = n
        Cells = 2**j
        msh = UnitSquareMesh(Cells, Cells, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=Cells, layer_height=1/(Cells))

        Sminus = FunctionSpace(mesh, "SminusDiv", PolyDegree)
        DPC = FunctionSpace(mesh, "DPC", PolyDegree - 1)
        W = Sminus * DPC
        #DQ = FunctionSpace(mesh, "DQ", PolyDegree - 1)
        #NCF = FunctionSpace(mesh, "NCF", PolyDegree)
        #W = NCF * DQ
        DOFs = W.dim()

        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

        x, y, z = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)*sin(pi*z)
        sigmaex = grad(uex)
        f = -div(grad(uex))

        a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
        apc = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx + inner(u, v)*dx

        L = -f*v*dx

        w = Function(W)

        params = {"mat_type": "matfree",
                    "pmat_type": "aij",
                    "ksp_type": "gmres",
                    "pc_type": "lu",
                    "ksp_monitor": None}

        solve(a == L, w, Jp=apc, solver_parameters=params)
        sigma, u = w.split()

        ErrVal = norms.errornorm(uex, u)
        SigErrVal = norms.errornorm(sigmaex, sigma)
        Info = [[PolyDegree], [Cells], [DOFs], [ErrVal], [SigErrVal]]
        file=open("3d-results-poisson-Sminus.csv", 'a', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(Info)
        print(ErrVal, SigErrVal)
