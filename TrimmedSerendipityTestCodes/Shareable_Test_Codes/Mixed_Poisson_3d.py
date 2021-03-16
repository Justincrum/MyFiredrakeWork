#Modified test code to work on Mixed Poisson problem.  Want to test Sminus Div elements in 2 and 3 dimensions.

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from firedrake.petsc import PETSc

parser = argparse.ArgumentParser(description="Allows for input of order and mesh refinement.")
parser.add_argument("-O", "--Order", type=int, help="Input the order of the polynomials.")
parser.add_argument("-S", "--Size", type=int, help="Input the exponent for number of cells of mesh 2**S.")
args = parser.parse_args()

for n in range(args.Order, args.Order + 1):
    for j in range(args.Size, args.Size + 1):
        PolyDegree = n
        Cells = 2**j
        msh = UnitSquareMesh(Cells, Cells, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=Cells, layer_height=1/(Cells))

        HDivSpace = FunctionSpace(mesh, "SminusDiv", PolyDegree)
        L2Space = FunctionSpace(mesh, "DPC", PolyDegree - 1)
        MixedSpace = HDivSpace * L2Space
        DOFs = MixedSpace.dim()

        sigma, u = TrialFunctions(MixedSpace)
        tau, v = TestFunctions(MixedSpace)

        x, y, z = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)*sin(pi*z)
        sigmaex = grad(uex)
        f = -div(grad(uex))

        a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
        L = -f*v*dx
        w = Function(MixedSpace)                                                         #Dummy function

        params = {"snes_type": "newtonls",
                  "snes_linesearch_type": "basic",
                  "snes_monitor": None,
                  "snes_converged_reason": None,
                  "mat_type": "aij",
                  "snes_max_it": 10,
                  "snes_lag_jacobian": -2,
                  "snes_lag_preconditioner": -2,
                  "ksp_type": "preonly",
                  "ksp_converged_reason": None,
                  "ksp_monitor_true_residual": None,
                  "pc_type": "lu",
                  "snes_rtol": 1e-12,
                  "snes_atol": 1e-20,
                  "pc_factor_mat_solver_type": "mumps",
                  "mat_mumps_icntl_14": "200",
                  "mat_mumps_icntl_11": "2"}
        PETSc.Log.begin()
        with PETSc.Log.Event("Solve"):
            solve(a == L, w, solver_parameters=params)
        Time = PETSc.Log.Event("Solve").getPerfInfo()["time"] 
        sigma, u = w.split()

        ErrVal = norms.errornorm(uex, u)                                                        #L2 Error between approximate u and exact u.
        SigErrVal = norms.errornorm(sigmaex, sigma)                                             #L2 Error between approximate sigma and exact sigma.
        Info = [PolyDegree, Cells, 1/Cells, DOFs, ErrVal, SigErrVal, Time]
        PETSc.Sys.Print(Info)
