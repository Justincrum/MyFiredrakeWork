#Modifying the test mixed poisson equation script to try and test it using different elements.
from firedrake import *
import numpy as np

""" size = 1

# Create mesh
mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=True)
x = SpatialCoordinate(mesh)

# Define function spaces and mixed (product) space
#BDM = FunctionSpace(mesh, "BDM" if not quadrilateral else "RTCF", 1)
BDM = FunctionSpace(mesh, "TBDMCE", 3)
DG = FunctionSpace(mesh, "DPC", 3)
W = BDM * DG

# Define trial and test functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

# Define source function
f = Function(DG).assign(0)
0
# Define variational form.
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
n = FacetNormal(mesh)
L = -f*v*dx + 42*dot(tau, n)*ds(4)

# Apply dot(sigma, n) == 0 on left and right boundaries strongly
# (corresponding to Neumann condition du/dn = 0)
bcs = DirichletBC(W.sub(0), Constant((0, 0)), (1, 2))
# Compute solution
w = Function(W)

solve(a == L, w, bcs=bcs, solver_parameters=parameters)
sigma, u = w.split()
print(u)


# Analytical solution
f.interpolate(42*x[1])
print(sqrt(assemble(dot(u - f, u - f) * dx)))
 """

size = 3

# Create mesh
mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=quadrilateral)
x = SpatialCoordinate(mesh)

    # Define function spaces and mixed (product) space
#BDM = FunctionSpace(mesh, "BDMC" if not quadrilateral else "RTCF", 1)
#BDM = FunctionSpace(mesh, "RTCF", 4)
BDMC = FunctionSpace(mesh, "BDMCF", 1)
#TBDMC = FunctionSpace(mesh, "TBDMCF", 3)
DPC = FunctionSpace(mesh, "DPC", 2)
#DG = FunctionSpace(mesh, "DG", 4)
#W = BDM * DG
W = BDMC * DPC

    # Define trial and test functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

    # Define source function
#f = Function(DG).assign(0)
f = Function(DPC).assign(0)

    # Define variational form
a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
n = FacetNormal(mesh)
L = -f*v*dx + 42*dot(tau, n)*ds(4)

    # Apply dot(sigma, n) == 0 on left and right boundaries strongly
    # (corresponding to Neumann condition du/dn = 0)
bcs = DirichletBC(W.sub(0), Constant((0, 0)), (1, 2))
    # Compute solution
w = Function(W)

solver_params = {'ksp_atol' : 1e-30, 'ksp_rtol' : 1e-16, 'ksp_divtol' : 1e7}

solve(a == L, w, bcs=[bcs], solver_parameters=solver_params)
sigma, u = w.split()

    # Analytical solution
f.interpolate(42*x[1])
print(sqrt(assemble(dot(u - f, u - f) * dx)))
