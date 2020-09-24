#Modified test code to work on Mixed Poisson problem.  Want to test Sminus Div elements in 2 and 3 dimensions.

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys

PolyDegree = 3
#mesh = UnitSquareMesh(8, 8)
msh = UnitSquareMesh(8, 8, quadrilateral=True)
mesh = ExtrudedMesh(msh, layers=8, layer_height=.125)
#Function spaces, currently testing against implemented spaces RTC * DQ
Sminus = FunctionSpace(mesh, "SminusDiv", PolyDegree)
DPC = FunctionSpace(mesh, "DPC", PolyDegree - 1)
#DQ = FunctionSpace(mesh, "DQ", PolyDegree - 1)
#NCF = FunctionSpace(mesh, "NCF", PolyDegree)
#W = NCF * DQ
W = Sminus * DPC

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
print(ErrVal, SigErrVal)