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
        
        H1Space = FunctionSpace(mesh, "S", PolyDegree) 
        DOFs = H1Space.dim()

        u = TrialFunction(H1Space)
        v = TestFunction(H1Space)

        x, y, z = SpatialCoordinate(mesh)
        wex = sin(pi*x)*sin(pi*y)*sin(pi*z)

        f = Function(H1Space)
        f = -div(grad(wex))

        a = dot(grad(u), grad(v))*dx
        L = inner(v,f)*dx

        bc1 = DirichletBC(H1Space, 0.0, "on_boundary")
        bct = DirichletBC(H1Space, 0.0, "top")
        bcb = DirichletBC(H1Space, 0.0, "bottom")
        w = Function(H1Space)                              #Dummy Function

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
                  "mat_mumps_icntl_14": "1000"}
        PETSc.Log.begin()
        with PETSc.Log.Event("Solve"):
            solve(a == L, w, bcs=[bc1,bct,bcb], solver_parameters=params)
        Time = PETSc.Log.Event("Solve").getPerfInfo()["time"]
        ErrVal = norms.errornorm(wex, w)                 #L2 Error between exact solution and computed solution.
        Info = [PolyDegree, Cells, DOFs, ErrVal, Time]
        PETSc.Sys.Print(Info)
