#Testing how to run an extruded mesh example.

from firedrake import *
m = UnitSquareMesh(20, 20, quadrilateral = True)
mesh = ExtrudedMesh(m, layers=10, layer_height=0.02)

V = FunctionSpace(mesh, "DPC", 1)

# RT1 element on a prism
""" W0_h = FiniteElement("RT", "triangle", 1)
W0_v = FiniteElement("DG", "interval", 0)
W0 = HDivElement(TensorProductElement(W0_h, W0_v))
W1_h = FiniteElement("DG", "triangle", 0)
W1_v = FiniteElement("CG", "interval", 1)
W1 = HDivElement(TensorProductElement(W1_h, W1_v))
W_elt = W0 + W1
W = FunctionSpace(mesh, W_elt) """

W = FunctionSpace(mesh, "SminusE", 2)

velocity = as_vector((0.0, 0.0, 1.0))
u = project(velocity, W)

x, y, z = SpatialCoordinate(mesh)
inflow = conditional(And(z < 0.02, x > 0.5), 1.0, -1.0)
q_in = Function(V)
q_in.interpolate(inflow)

n = FacetNormal(mesh)
un = 0.5*(dot(u, n) + abs(dot(u, n)))

q = TrialFunction(V)
phi = TestFunction(V)

a1 = -q*dot(u, grad(phi))*dx
a2 = dot(jump(phi), un('+')*q('+') - un('-')*q('-'))*dS_h
a3 = dot(phi, un*q)*ds_t  # outflow at top wall
a = a1 + a2 + a3

L = -q_in*phi*dot(u, n)*ds_b  # inflow at bottom wall

out = Function(V)
solve(a == L, out)

exact = Function(V)
exact.interpolate(conditional(x > 0.5, 1.0, -1.0))

print(max(abs(out.dat.data - exact.dat.data)))

assert max(abs(out.dat.data - exact.dat.data)) < 1e-10
