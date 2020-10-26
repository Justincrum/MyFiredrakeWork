#Test code to try Lawrence's suggestion.
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import csv
#ErrorValues = []
#GridSize = []

Data = []
for p in range(2, 8):
    ErrorValues = []
    RateValues = []
    GridSize = []
    PolyDegrees = []
    NCells = []
    DOFS = []
    for j in range(2, 8):
        PolyDegree = p
        Cells = 2**j
        h = 1 / Cells
        mesh = UnitSquareMesh(Cells, Cells, quadrilateral=True)
        Sminus = FunctionSpace(mesh, "SminusCurl", PolyDegree)
        DOFs = Sminus.dim()
        x, y = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)
        sigmaex = grad(uex)
        params = {"snes_type": "newtonls", "snes_max_it": "2", "snes_convergence_test": "skip", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        #err = errornorm(uex, project(uex, S))
        err = errornorm(sigmaex, project(sigmaex, Sminus, solver_parameters=params))
        #err = errornorm(sigmaex, project(sigmaex, NCF))
        ErrorValues.append(err)
        GridSize.append(h)
        PolyDegrees.append(PolyDegree)
        NCells.append(Cells)
        DOFS.append(DOFs)
        CurrentData = [PolyDegree, Cells, h, DOFs, err]
        Data.append(CurrentData)
    #print(ErrorValues)
    for j in range(0, len(ErrorValues)-1):
        top = np.log(ErrorValues[j]/ErrorValues[j+1])
        bottom = np.log(GridSize[j] / GridSize[j+1])
        rate = top / bottom
        RateValues.append([rate])
    #print(RateValues)
    #Info = [PolyDegrees, NCells, GridSize, DOFS, ErrorValues]
file=open("2d-projection-SminusCurl.csv", 'a', newline = '')
with file:
    write = csv.writer(file, delimiter=',')
    write.writerows(Data)
        #write.writerows(RateValues)    
