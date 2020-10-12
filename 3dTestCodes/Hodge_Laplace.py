#Solving the Hodge Laplacian on a unit box and unit cube.
#The equations of interest are inner(tau, sigma) - inner(curl tau, u) + inn(v, curl sigma) + inner(div v, div u) = inner(v, f)
#The exact solution of this is u(x, y, z) = ( x^2(x-1)^2 * sin(pi * y) * sin(pi * z), y^2(y-1)^2 sin(pi * x) * sin(pi * z), z^2(z-1)^2 sin(pi *x) * sin(pi * y))

from firedrake import *
m = UnitSquareMesh(20, 20, quadrilateral = True)
mesh = ExtrudedMesh(m, layers=10, layer_height=0.02)

degree = 3

Curl = FunctionSpace(mesh, "Nedelec", degree)
Div = FunctionSpace(mesh, "DQ", degree - 1)

W = Curl * Div
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)



