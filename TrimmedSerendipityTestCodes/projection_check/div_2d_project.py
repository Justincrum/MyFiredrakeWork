#Test code to try Lawrence's suggestion.
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
#ErrorValues = []
#GridSize = []
for p in range(5, 6):
    ErrorValues = []
    RateValues = []
    GridSize = []
    for j in range(2, 9):
        PolyDegree = p
        Cells = 2**j
        h = 1 / Cells
        mesh = UnitSquareMesh(Cells, Cells, quadrilateral=True)
        #mesh = ExtrudedMesh(msh, layers=Cells, layer_height=1/(Cells))
        Sminus = FunctionSpace(mesh, "SminusDiv", PolyDegree)
        #BDMC = FunctionSpace(mesh, "BDMCF", PolyDegree)
        #NCF = FunctionSpace(mesh, "RTCF", PolyDegree)
        #S = FunctionSpace(mesh, "S", PolyDegree)
        x, y = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)
        sigmaex = grad(uex)
        params = {"snes_type": "newtonls", "snes_max_it": "2", "snes_convergence_test": "skip", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        #err = errornorm(uex, project(uex, S))
        err = errornorm(sigmaex, project(sigmaex, Sminus, solver_parameters=params))
        #err = errornorm(sigmaex, project(sigmaex, NCF))
        ErrorValues.append(err)
        GridSize.append(h)
    print(ErrorValues)
    for j in range(0, len(ErrorValues)-1):
        top = np.log(ErrorValues[j]/ErrorValues[j+1])
        bottom = np.log(GridSize[j] / GridSize[j+1])
        rate = top / bottom
        RateValues.append(rate)
    print(RateValues)
        #print(errornorm(sigmaex, project(sigmaex, Sminus)))
        #F = project(sigmaex, Sminus)
        #print(F.dat.data_ro)
    
