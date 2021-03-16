from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from firedrake.petsc import PETSc

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

        PickASpace = FunctionSpace(mesh, "SminusCurl", PolyDegree) #NCE = Q^- Lambda^1, NCF = Q^- Lambda^2, SminusCurl = S^- Lambda^1, SminusDiv = S^- Lambda^2
        DOFs = PickASpace.dim()

        x, y, z = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)*sin(pi*z)
        sigmaex = grad(uex)

        params={"mat_type": "aij", "ksp_type": "cg", "pc_type": "bjacobi", "sub_pc_type": "ilu", "ksp_rtol": 1e-10}
        
        err = errornorm(sigmaex, project(sigmaex, PickASpace, solver_parameters=params))
        
        ErrorValues.append(err)
        GridSize.append(h)
        PolyDegrees.append(PolyDegree)
        NCells.append(Cells)
        DOFS.append(DOFs)
        CurrentData = [PolyDegree, Cells, h, DOFs, err]
        Data.append(CurrentData)
        
PETSc.Sys.Print(Data)
