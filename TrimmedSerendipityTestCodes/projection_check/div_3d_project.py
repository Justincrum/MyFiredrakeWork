#Test code to try Lawrence's suggestion.
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

ErrorValues = []
GridSize = []
Data = []
for p in range(2, 3):
    ErrorValues = []
    RateValues = []
    GridSize = []
    PolyDegrees = []
    NCells = []
    DOFS = []
    for j in range(3, 4):
        PolyDegree = p
        Cells = 2**j
        h = 1 / Cells
        msh = UnitSquareMesh(Cells, Cells, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=Cells, layer_height=1/(Cells))

        Sminus = FunctionSpace(mesh, "SminusCurl", PolyDegree) #NCE for tensor product 1-forms.
        DOFs = Sminus.dim()
        x, y, z = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)*sin(pi*z)
        #uex = e**x*sin(pi*y)*z
        sigmaex = grad(uex)
        #err = errornorm(sigmaex, project(sigmaex, NCF))
        params = {"mat_type": "aij", "snes_type": "newtonls", "snes_max_it": "2", "snes_convergence_test": "skip", "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        #params = {"mat_type":"mat_free", "snes_type": "newtonls", "snes_max_it": "2", "snes_convergence_test": "skip", "ksp_type": "gmres", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": "200"}
        err = errornorm(sigmaex, project(sigmaex, Sminus, solver_parameters=params))
        ErrorValues.append(err)
        GridSize.append(h)
        PolyDegrees.append(PolyDegree)
        NCells.append(Cells)
        DOFS.append(DOFs)
        CurrentData = [PolyDegree, Cells, h, DOFs, err]
        Data.append(CurrentData)
    print(ErrorValues)
    for j in range(0, len(ErrorValues)-1):
        top = np.log(ErrorValues[j]/ErrorValues[j+1])
        bottom = np.log(GridSize[j] / GridSize[j+1])
        rate = top / bottom
        RateValues.append(rate)
    print(RateValues)
file=open("3d-projection-SminusCurl.csv", 'a', newline = '')
with file:
    write = csv.writer(file, delimiter=',')
    write.writerows(Data)
