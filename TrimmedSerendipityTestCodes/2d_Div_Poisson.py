#Modified test code to work on Mixed Poisson problem.  Want to test Sminus Div elements in 2 and 3 dimensions.

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
for n in range(1, 6):
    for j in range(3, 6):
        PolyDegree = n
        #mesh = UnitSquareMesh(8, 8)
        Cells = 2**j
        mesh = UnitSquareMesh(Cells, Cells, quadrilateral=True)

        #Function spaces, currently testing against implemented spaces RTC * DQ
        Sminus = FunctionSpace(mesh, "SminusDiv", PolyDegree)
        DPC = FunctionSpace(mesh, "DPC", PolyDegree - 1)
        #DQ = FunctionSpace(mesh, "DQ", PolyDegree - 1)
        #RTC = FunctionSpace(mesh, "RTCF", PolyDegree)
        #BDM = FunctionSpace(mesh, "BDM", PolyDegree)
        #DG = FunctionSpace(mesh, "DG", PolyDegree - 1)
        #W = BDM * DG
        #W = RTC * DQ
        W = Sminus * DPC

        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

        x, y = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)
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

        file=open("2d_Div_Mixed_poisson.csv", 'a', newline='')
        with file:
            write = csv.writer(file)
            write.writerows('Degree of element')
            write.writerows(PolyDegree)
            write.writerows('Number of Cells in x and y')
            write.writerows(Cells)
            write.writerows('Error value in u')
            write.writerows(ErrVal)
            write.writerows('Error value in sigma')
            write.writerows(SigErrVal)
            write.writerows(' ')
#print(ErrVal, SigErrVal)
