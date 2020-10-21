#Attempt at checking projection with hybridization parameters.
from firedrake import *
import numpy as np
base_mesh = UnitSquareMesh(8, 8, quadrilateral=True)
MH = MeshHierarchy(base_mesh, 4)
#EMH = ExtrudedMeshHierarchy(MH, 1, base_layer=4)
for deg in range(2, 3):
    for msh in MH:
        Sminus = FunctionSpace(msh, "SminusDiv", deg)  #Substitute this out for NCF as desired.
        x, y = SpatialCoordinate(msh)
        uex = sin(pi*x)*sin(pi*y)
        sigmaex = grad(uex)
        """ hybrid_params = {'mat_type': 'matfree',
                         'ksp_type': 'preonly',
                         'pc_type': 'python',
                         'pc_python_type': 'firedrake.HybridizationPC',
                         'hybridization': {'ksp_type': 'preonly',
                                           'ksp_gmres_restart': 1,
                                           'pc_type': 'lu'}} """
        params = {"snes_type": "newtonls", "snes_max_it": "3", "snes_convergence_test": "skip", "ksp_type": "preonly","ksp_monitor":None, "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        err = errornorm(sigmaex, project(sigmaex, Sminus, solver_parameters=params))
        print(err)
