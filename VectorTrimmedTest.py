"""Solve the mixed formulation of the Laplacian on the unit square
    sigma - grad(u) = 0
         div(sigma) = f
The corresponding weak (variational problem)
    <sigma, tau> + <div(tau), u>   = 0       for all tau
                   <div(sigma), v> = <f, v>  for all v
is solved using BDM (Brezzi-Douglas-Marini) elements of degree k for
(sigma, tau) and DG (discontinuous Galerkin) elements of degree k - 1
for (u, v).
The boundary conditions on the left and right are enforced strongly as
    dot(sigma, n) = 0
which corresponds to a Neumann condition du/dn = 0.
The top is fixed to 42 with a Dirichlet boundary condition, which enters
the weak formulation of the right hand side as
    42*dot(tau, n)*ds
"""

from firedrake import *

size = 2
# Create mesh
mesh = UnitSquareMesh(2 ** size, 2 ** size, quadrilateral=quadrilateral)
x = SpatialCoordinate(mesh)
# Define function spaces and mixed (product) space
###BDM = FunctionSpace(mesh, "BDM" if not quadrilateral else "RTCF", 1)
BDM = FunctionSpace(mesh, "TBDMCE", 2)
###DG = FunctionSpace(mesh, "DG", 0)
DG = FunctionSpace(mesh, "DPC", 2)
W = BDM * DG

# Define trial and test functions
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

# Define source function
f = Function(DG).assign(0)

# Define variational form
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

# Analytical solution
f.interpolate(42*x[1])
print(sqrt(assemble(dot(u - f, u - f) * dx)))
