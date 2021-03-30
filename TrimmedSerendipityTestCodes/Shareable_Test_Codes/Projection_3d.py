#
#file: Projection_3d.py
#author:  Justin Crum
#date:  3/19/21
#
"""
Copyright 2021 Justin Crum

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
from firedrake import *
import argparse

parser = argparse.ArgumentParser(
        description="Allows for input of order and mesh refinement.")
parser.add_argument("-O", "--Order", 
        type=int, help="Input the order of the polynomials.")
parser.add_argument("-N", "--CellExponent", 
        type=int, help="Input the exponent for number of cells in mesh 2**N x 2**N x 2**N.")
args = parser.parse_args()

for n in range(args.Order, args.Order + 1):
    for j in range(args.CellExponent, args.CellExponent + 1):

        ###Setting up mesh.
        polyDegree = n
        numberOfCells = 2**j
        h = 1 / numberOfCells
        msh = UnitSquareMesh(numberOfCells, 
                             numberOfCells, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=numberOfCells, 
                            layer_height=1/(numberOfCells))

        ###Choosing function space.
        #Could use NCE, NCF, Lagrange, SminusCurl, SminusDiv, DPC.
        pickASpace = FunctionSpace(mesh, "SminusCurl", polyDegree)
        dofs = pickASpace.dim()

        ###Setting up problem, solver parameters, solving, and data collection.
        x, y, z = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)*sin(pi*z)
        sigmaex = grad(uex)

        ###Solver parameters and solving.
        params={"mat_type": "aij", 
                "ksp_type": "cg", 
                "pc_type": "bjacobi", 
                "sub_pc_type": "ilu", 
                "ksp_rtol": 1e-10}
        
        err = errornorm(sigmaex, project(sigmaex, pickASpace, 
                        solver_parameters=params))
        currentData = [polyDegree, numberOfCells, h, dofs, err]
        print(currentData)