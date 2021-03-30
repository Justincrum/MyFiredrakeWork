#
#file: Projection_2d.py
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
        type=int, help="Input the exponent for number of cells in mesh 2**N x 2**N.")
args = parser.parse_args()

for n in range(args.Order, args.Order + 1):
    for j in range(args.CellExponent, args.CellExponent + 1):

        ###Setting up the mesh.
        polyDegree = n
        numberOfCells = 2**j
        h = 1 / numberOfCells
        mesh = UnitSquareMesh(numberOfCells, numberOfCells, 
                              quadrilateral=True)
        
        ###Choose function space and check the DOFs count.  
        #Can replace SminusCurl with SminusDiv, RTCE, RTCF, S, or Lagrange.  
        #For S and Lagrange, change to the scalar element projection line.
        pickASpace = FunctionSpace(mesh, "SminusCurl", polyDegree)
        dofs = pickASpace.dim()
        
        ###Solve the problem.
        x, y = SpatialCoordinate(mesh)
        uex = sin(pi*x)*sin(pi*y)
        sigmaex = grad(uex)
        params={"mat_type": "aij", 
                "ksp_type": "cg", 
                "pc_type": "bjacobi", 
                "sub_pc_type": "ilu", 
                "ksp_rtol": 1e-10}
        #For scalar elements.
        #err = errornorm(uex, project(uex, pickASpace, 
        #                solver_parameters=params))
        err = errornorm(sigmaex, project(sigmaex, pickASpace, 
                        solver_parameters=params)) #For vector elements.
        currentData = [polyDegree, numberOfCells, h, dofs, err]
    print(currentData)
