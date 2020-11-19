# example usage
# python chladni.py -eps_nev 5 -eps_tol 1e-11 -eps_target 10 -st_type sinvert
from firedrake import *
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import csv
OverallEigenvalueList = []
iterationCount = []
for deg in range(4, 5):
    for j in range(2, 6):
        N = 2 ** j
        msh = RectangleMesh(N, N, np.pi, np.pi, quadrilateral=True)
        mesh = ExtrudedMesh(msh, layers=N, layer_height=np.pi/(N)) 
        # Some finer meshes for visualisation
        # deg = 1
        #V = FunctionSpace(mesh, "SminusCurl", deg)
        V = FunctionSpace(mesh, "NCE", deg)
        u = TrialFunction(V)
        v = TestFunction(V)
        PETSc.Sys.Print("DoFs of V are:")
        PETSc.Sys.Print(V.dim())
                
        a = (inner(curl(u), curl(v)))*dx
        mss = inner(u, v)*dx
        bc=DirichletBC(V, 0, "on_boundary")
        # This prefix allows us to control factorisation package from options.
        A = assemble(a, bcs=bc, options_prefix="st_").M.handle
        M = assemble(mss, bcs=bc, options_prefix="st_").M.handle
        E = SLEPc.EPS().create(comm=msh.comm)
        E.setOperators(A, M)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        # Defaults  -eps_nev 5 -eps_tol 1e-11 -eps_target 10 -st_type sinvert
        E.setTolerances(tol=1e-11)
        E.setTarget(1.5)
        E.setDimensions(nev=15)
        E.st.setType(SLEPc.ST.Type.SINVERT)
        E.setFromOptions()
        E.solve()
        Print = PETSc.Sys.Print
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
                #    a = 4
                    Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                else:
                #    a = 2
                    Print(" %12f      %12g" % (k.real, error))
                #EigenList += [(k.real, error)]
                OverallEigenvalueList.append([k.real, error])
    file=open('Pi_NCE_Eigenvalue_Results.csv', 'a', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(OverallEigenvalueList)
#        for j in range(0, 5):
#            L = [('This is trimmed Serendipity with degree', deg, 'and N =', 2 ** (j+1))]
#            write.writerows(L)
#            write.writerows(OverallEigenvalueList[j])
#            write.writerows(' ')
#file = open('TrimmedS_Eigenvalue_Results.csv', 'a', newline='')
#with file:
#    write=csv.writer(file)
#    write.writerows(iterationCount)
