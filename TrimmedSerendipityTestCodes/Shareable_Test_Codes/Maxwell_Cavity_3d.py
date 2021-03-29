#file:  Maxwell_Cavity_3d.py
#author:  Justin Crum
#date:  3/19/21
#
#
# This code does an eigenvalue solve using Firedrake
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
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

OverallEigenvalueList = []
iterationCount = []

for polyDegree in range(2, 3):            
    for j in range(3, 4):     

        ###Mesh set up.
        N = 2 ** j
        msh = RectangleMesh(N, N, np.pi, np.pi, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=N, layer_height=np.pi/(N)) 

        ###Function Space set up.
        #The FunctionSpace call here could use SminusCurl or NCE.
        hCurlSpace = FunctionSpace(mesh, "SminusCurl", polyDegree)
        u = TrialFunction(hCurlSpace)
        v = TestFunction(hCurlSpace)
        PETSc.Sys.Print("DoFs of hCurlSpace are:")
        PETSc.Sys.Print(hCurlSpace.dim())

        ###Problem set up.
        a = (inner(curl(u), curl(v)))*dx
        mss = inner(u, v)*dx
        bc = DirichletBC(hCurlSpace, 0, "on_boundary")
        bct = DirichletBC(hCurlSpace, 0.0, "top")
        bcb = DirichletBC(hCurlSpace, 0.0, "bottom")

        ###Eigensolver set up.
        A = assemble(a, bcs=[bc,bct,bcb], 
                     options_prefix="st_").M.handle
        M = assemble(mss, bcs=[bc,bct,bcb], 
                     options_prefix="st_").M.handle
        E = SLEPc.EPS().create(comm=mesh.comm)
        E.setOperators(A, M)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        # Defaults  -eps_nev 5 -eps_tol 1e-11 -eps_target 10 -st_type sinvert
        E.setTolerances(tol=1e-11)
        E.setDimensions(nev=15)
        E.st.setType(SLEPc.ST.Type.SINVERT)
        E.setTarget(3.0)
        E.setFromOptions()
        E.setUp()

        ###SLEPc solve and print out of results.
        PETSc.Log.begin()
        with PETSc.Log.Event("Solve"): 
            E.solve()
        Time = PETSc.Log.Event("Solve").getPerfInfo()["time"]
        Print = PETSc.Sys.Print
        Print("SLEPc solve time: %f" % Time)
        Print()
        Print("******************************")
        Print("*** SLEPc Solution Results for SminusCurl ***")
        Print("******************************")
        Print()
        its = E.getIterationNumber()
        Print("Number of iterations of the method: %d" % its)
        iterationCount += [[(its)]]
        eps_type = E.getType()
        Print("Solution method: %s" % eps_type)
        nev, ncv, mpd = E.getDimensions()
        Print("Number of requested eigenvalues: %d" % nev)
        tol, maxit = E.getTolerances()
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        nconv = E.getConverged()
        Print("Number of converged eigenpairs %d" % nconv)

        ###Data collection
        if nconv > 0:
            vr, wr = A.getVecs()
            vi, wi = A.getVecs()
            ks = []
            for i in range(nconv):
                k = E.getEigenpair(i, vr, vi)
                ks.append(k.real)
                inds = np.argsort(ks)
        if nconv > 0:

        ###Create the results vectors
            vr, wr = A.getVecs()
            vi, wi = A.getVecs()
            EigenList = []
            for j, i in enumerate(inds):
                k = E.getEigenpair(i, vr, vi)
                error = E.computeError(i)
                if k.real < 1e-10:
                    continue
                if k.imag != 0.0:
                    Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                else:
                    Print(" %12f      %12g" % (k.real, error))
                OverallEigenvalueList.append([k.real, error])
