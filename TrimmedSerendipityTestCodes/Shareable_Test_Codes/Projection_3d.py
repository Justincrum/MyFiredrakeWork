#
#file: Projection_3d.py
#author:  Justin Crum
#date:  3/19/21
#
"""
Copyright 2021 Justin Crum

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
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
