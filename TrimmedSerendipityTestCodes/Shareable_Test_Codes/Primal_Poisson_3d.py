#
#file:  Primal_Poisson_3d.py
#author:  Justin Crum
#date:  3/19/21
#
"""
Copyright 2021 Justin Crum

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
from firedrake import *
import argparse
from firedrake.petsc import PETSc

parser = argparse.ArgumentParser(
        description="Allows for input of order and mesh refinement.")
parser.add_argument("-O", "--Order", 
        type=int, help="Input the order of the polynomials.")
parser.add_argument("-S", "--Size", 
        type=int, help="Input the exponent for number of cells of mesh 2**S.")
args = parser.parse_args()

for n in range(args.Order, args.Order + 1):
    for j in range(args.Size, args.Size + 1):

        ###Mesh set up
        polyDegree = n
        numberOfCells = 2**j
        msh = UnitSquareMesh(numberOfCells, numberOfCells, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=numberOfCells, 
                            layer_height=1/(numberOfCells))

        ###Function space set up.
        #Could use S or Lagrange.
        h1Space = FunctionSpace(mesh, "S", polyDegree) 
        dofs = h1Space.dim()

        u = TrialFunction(h1Space)
        v = TestFunction(h1Space)

        ###Problem set up.
        x, y, z = SpatialCoordinate(mesh)
        wex = sin(pi*x)*sin(pi*y)*sin(pi*z)

        f = Function(h1Space)
        f = -div(grad(wex))

        a = dot(grad(u), grad(v))*dx
        l = inner(v,f)*dx

        bc1 = DirichletBC(h1Space, 0.0, "on_boundary")
        bct = DirichletBC(h1Space, 0.0, "top")
        bcb = DirichletBC(h1Space, 0.0, "bottom")
        w = Function(h1Space)

        ###Solver parameters, solving, and printing results.
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
            solve(a == l, w, bcs=[bc1,bct,bcb], solver_parameters=params)
        time = PETSc.Log.Event("Solve").getPerfInfo()["time"]
        errVal = norms.errornorm(wex, w)
        info = [polyDegree, numberOfCells, dofs, errVal, time]
        PETSc.Sys.Print(info)
