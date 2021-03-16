from firedrake import *
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import csv

OverallEigenvalueList = []
iterationCount = []

for PolyDegree in range(2, 3):            
    for j in range(3, 4):          #This determines the size of your mesh.
        N = 2 ** j
        msh = RectangleMesh(N, N, np.pi, np.pi, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=N, layer_height=np.pi/(N)) 

        HCurlSpace = FunctionSpace(mesh, "SminusCurl", PolyDegree)
        u = TrialFunction(HCurlSpace)
        v = TestFunction(HCurlSpace)
        PETSc.Sys.Print("DoFs of HCurlSpace are:")
        PETSc.Sys.Print(HCurlSpace.dim())
                
        a = (inner(curl(u), curl(v)))*dx
        mss = inner(u, v)*dx
        bc=DirichletBC(HCurlSpace, 0, "on_boundary")
        bct = DirichletBC(HCurlSpace, 0.0, "top")
        bcb = DirichletBC(HCurlSpace, 0.0, "bottom")
        
        A = assemble(a, bcs=[bc,bct,bcb], options_prefix="st_").M.handle
        M = assemble(mss, bcs=[bc,bct,bcb], options_prefix="st_").M.handle
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

        if nconv > 0:
            vr, wr = A.getVecs()
            vi, wi = A.getVecs()
            ks = []
            for i in range(nconv):
                k = E.getEigenpair(i, vr, vi)
                ks.append(k.real)
                inds = np.argsort(ks)
        if nconv > 0:
        # Create the results vectors
            vr, wr = A.getVecs()
            vi, wi = A.getVecs()
            #
            #Print()
            #Print("        k          ||Ax-kx||/||kx||")
            #Print("----------------- ------------------")
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
