#Recreating mixed_poison.py for testing SminusDiv elements.

from firedrake import *
import numpy as np

N = 8
mesh = UnitSquareMesh(N, N)
#mesh = UnitSquareMesh(N, N, quadrilateral=True) 

deg = 1
BDM = FunctionSpace(mesh, "BDM", deg)
DG = FunctionSpace(mesh, "DG", deg - 1)
W = BDM * DG

#Sminus = FunctionSpace(mesh, "SminusDiv", deg)
#DPC = FunctionSpace(mesh, "DPC", deg - 1)
#W = Sminus * DPC

sigma, u, = TrialFunctions(W)
tau, v = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
#f = Function(DG).project(2 * pi**2 * sin(pi * x) * sin(pi * y))
uex = sin(pi*x) * sin(pi*y)
f = div(grad(uex))

a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
L = - f*v*dx

bc = DirichletBC(W.sub(0), 0, "on_boundary") 

w = Function(W)

params = {"pmat_type": "aij",
          "ksp_type": "gmres",
          "pc_type": "lu",
          "ksp_monitor": None}

solve(a == L, w, bcs=bc, solver_parameters=params)
sigma, u = w.split()

print(errornorm(uex, u))

