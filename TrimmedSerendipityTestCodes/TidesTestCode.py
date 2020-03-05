#Code given by Rob Kirby used to test my implementation of trimmed serendipity code.


from firedrake import *


N = 32
mesh = UnitSquareMesh(N, N, quadrilateral=True)
# mh = MeshHierarchy(base_msh, 0)
# mesh = mh[-1]

BDM = FunctionSpace(mesh, "SminusE", 2)
DG = FunctionSpace(mesh, "DPC", 1)
W = BDM * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

x, y = SpatialCoordinate(mesh)
uex = sin(pi*x)*sin(pi*y)
f = -curl(curl(uex))

a = (dot(sigma, tau) + curl(tau)*u + curl(sigma)*v)*dx
apc = inner(sigma, tau)*dx + inner(curl(sigma), curl(tau))*dx + inner(u, v)*dx

L = f*v*dx

w = Function(W)

class Riesz(AuxiliaryOperatorPC):
  def form(self, pc, test, trial):
    a = inner(test, trial)*dx + inner(curl(test), curl(trial))*dx
    return (a, None)


# params = {"mat_type": "matfree",
#           "pmat_type": "matfree",
#           "ksp_type": "minres",
#           "ksp_monitor": None,
#           "pc_type": "fieldsplit",
#           "pc_fieldsplit_type": "additive",
#           "fieldsplit_0":{
#             #"ksp_type": "cg",
#             #"ksp_max_it": 100,
#             #"ksp_rtol": 1.0e-10,
#             #"ksp_atol": 0.0,
#             #"ksp_norm_type": "unpreconditioned",
#             #"ksp_monitor_true_residual": None,
#             "ksp_type": "preonly",
#             "pc_type": "python",
#             "pc_python_type": "__main__.Riesz",
#             "aux": {
#               "pc_type": "mg",
#               "pc_mg_type": "full",
#               "mg_levels": {
#                 "ksp_type": "richardson",
#                 "ksp_norm_type": "unpreconditioned",
#                 #"ksp_monitor_true_residual": None,
#                 "ksp_richardson_scale": 1/3,
#                 "ksp_max_it": 1,
#                 "ksp_convergence_test": "skip",
#                 "pc_type": "python",
#                 "pc_python_type": "firedrake.PatchPC",
#                 "patch_pc_patch_save_operators": True,
#                 "patch_pc_patch_partition_of_unity": False,
#                 "patch_pc_patch_construct_type": "star",
#                 "patch_pc_patch_construct_dim": 0,
#                 "patch_pc_patch_sub_mat_type": "seqdense",
#                 "patch_sub_ksp_type": "preonly",
#                 "patch_sub_pc_type": "lu"},
#               "mg_coarse_pc_type": "python",
#               "mg_coarse_pc_python_type": "firedrake.AssembledPC",
#               "mg_coarse_assembled_pc_type": "lu",
#               "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
#             },
#           },
#           "fieldsplit_1":{
#             "ksp_type": "preonly",
#             "pc_type": "python",
#             "pc_python_type": "firedrake.AssembledPC",
#             "assembled_pc_type": "jacobi"}          
#           }

params = {"mat_type": "matfree",
          "pmat_type": "aij",
          "ksp_type": "gmres",
          "pc_type": "lu",
          "ksp_monitor": None}

solve(a == L, w, Jp=apc, solver_parameters=params)
sigma, u = w.split()

print(errornorm(uex, u))
import matplotlib.pyplot as plt
plot(u)
plt.show()
