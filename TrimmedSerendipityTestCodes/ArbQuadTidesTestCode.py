#Code given by Rob Kirby used to test my implementation of trimmed serendipity code.


from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import sys

PolyDegree = int(sys.argv[1])
UErrors = []
SigErrors = []
CellCount = []
DofCount = []
Times = int(sys.argv[2])

for i in range (2, Times + 3):

    #Setting up the mesh to work on.
    Cells = 2**(i)
    #Setting up the mesh.
    nx = Cells
    ny = Cells
    Lx = 1
    Ly = 1
    mesh = utility_meshes.RectangleMesh(nx, ny, Lx, Ly, quadrilateral = True)

    #Now we'll take the unit square mesh and remodel it so that it is the
    #trapezoid mesh that we're interested in.
    for j in range(len(mesh.coordinates.dat.data)):
        X = mesh.coordinates.dat.data[j][0]
        Y = mesh.coordinates.dat.data[j][1]
        if(i == 1):
            PowerY = 10 * Y
            PowerX = 10 * X
        if(i == 2):
            PowerY = (10**2) * Y
            PowerX = (10**2) * X
        if(i == 3):
            PowerY = (10**3) * Y
            PowerX = (10**3) * X
        if(i == 4):
            PowerY = (10**4) * Y
            PowerX = (10**4) * X
        if(i == 5):
            PowerY = (10**5) * Y
            PowerX = (10**5) * X
        if(i == 6):
            PowerY = (10**6) * Y
            PowerX = (10**6) * X
        if(PowerY % 2 == 1):
            if(PowerX % 2 == 0):
                Y += -(1.0 / (2.0 * Cells))
                mesh.coordinates.dat.data[j,1] = Y
            if(PowerX % 2 == 1):
                Y += (1.0 / (2.0 * Cells))
                mesh.coordinates.dat.data[j,1] = Y

    #Testing out using trimmed serendipity space.
    #Sminus = FunctionSpace(mesh, "SminusE", PolyDegree)
    RTC = FunctionSpace(mesh, "RTCE", PolyDegree)
    #DPC = FunctionSpace(mesh, "DPC", PolyDegree)
    DQ = FunctionSpace(mesh, "DQ", PolyDegree)
    #W = Sminus * DPC
    W = RTC * DQ
    Dofs = W.dim()
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    uex = sin(pi*x)*sin(pi*y)
    sigmaex = -curl(uex)
    f = -curl(curl(uex))

    a = (dot(sigma, tau) + curl(tau)*u + curl(sigma)*v)*dx
    apc = inner(sigma, tau)*dx + inner(curl(sigma), curl(tau))*dx + inner(u, v)*dx

    L = f*v*dx

    w = Function(W)

    class Riesz(AuxiliaryOperatorPC):
      def form(self, pc, test, trial):
        a = inner(test, trial)*dx + inner(curl(test), curl(trial))*dx
        return (a, None)


    params = {"mat_type": "matfree",
              "pmat_type": "aij",
              "ksp_type": "gmres",
              "pc_type": "lu",
              "ksp_monitor": None}

    solve(a == L, w, Jp=apc, solver_parameters=params)
    sigma, u = w.split()

    ErrVal = norms.errornorm(uex, u)
    SigErrVal = norms.errornorm(sigmaex, sigma)
    #print(ErrVal, SigErrVal)

    UErrors.append(ErrVal)
    SigErrors.append(SigErrVal)
    CellCount.append(Cells * Cells)
    DofCount.append(Dofs)


UErrors = np.array(UErrors)
SigErrors = np.array(SigErrors)
Leng = np.max(np.shape(UErrors))
Rates = np.zeros([Leng - 1, 1])
SigRates = np.zeros([Leng - 1, 1])
SharedDofs = np.zeros([Leng, 1])
BoundaryDofs = np.zeros([Leng, 1])
NonbdDofs = np.zeros([Leng, 1])
for i in range(0, Times):
    h1 = 1.0 / CellCount[i]
    h2 = 1.0 / CellCount[i+1]
    Rates[i] = np.log2(UErrors[i] / UErrors[i+1])
    SigRates[i] = np.log2(SigErrors[i] / SigErrors[i+1])
    CellsAcross = int(np.sqrt(CellCount[i]))

#for i in range(0, Times + 1):
#    CellsAcross = int(np.sqrt(CellCount[i]))
#    SharedDofs[i] = (PolyDegree + 1) * (2 * CellsAcross * CellsAcross - 2 * CellsAcross)
#    BoundaryDofs[i] = (PolyDegree + 1) * ( CellsAcross * 4)
#    NonbdDofs[i] = DofCount[i] - SharedDofs[i] - BoundaryDofs[i]

for i in range(0, Times + 1):
   CellsAcross = int(np.sqrt(CellCount[i]))
   SharedDofs[i] = (PolyDegree) * (2 * CellsAcross * CellsAcross - 2 * CellsAcross)
   BoundaryDofs[i] = (PolyDegree) * (CellsAcross * 4)
   NonbdDofs[i] = DofCount[i] - SharedDofs[i] - BoundaryDofs[i]



from tabulate import tabulate
print("Degree:", PolyDegree, file=open("Comparison_output.txt", "a"))
print("RTC", file=open("Comparison_output.txt", "a"))
table = [[CellCount[k], UErrors[k], Rates[k-1], SigErrors[k], SigRates[k-1], DofCount[k], 
        SharedDofs[k], NonbdDofs[k]] for k in range(1, Times + 1)]
headers = ['Cells' , 'UError', 'URate', 'SigError', 'SigRate', 'Total DoFs', 'Shared DoFs', 'Nonboundary, nonshared DoFs']
print(tabulate(table, headers), file=open("Comparison_output.txt", "a"))
print("Iteration done!", file=open("Comparison_output.txt", "a"))
