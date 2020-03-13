#Code given by Rob Kirby used to test my implementation of trimmed serendipity code.


from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

PolyDegree = 3
UErrors = []
SigErrors = []
CellCount = []
Times = 3

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
    Sminus = FunctionSpace(mesh, "SminusE", 3)
    DPC = FunctionSpace(mesh, "DPC", 2)
    W = Sminus * DPC

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
    #DivErr = norms.errornorm(div(sigmaex), div(sigma))
    print(div(sigmaex))
    print(DivErr)
    #print(ErrVal, SigErrVal)

    UErrors.append(ErrVal)
    SigErrors.append(SigErrVal)
    CellCount.append(Cells)


UErrors = np.array(UErrors)
SigErrors = np.array(SigErrors)
Leng = np.max(np.shape(UErrors))
Rates = np.zeros([Leng - 1, 1])
SigRates = np.zeros([Leng - 1, 1])
for i in range(0, Times):
    h1 = 1.0 / CellCount[i]
    h2 = 1.0 / CellCount[i+1]
    Rates[i] = np.log2(UErrors[i] / UErrors[i+1])
    SigRates[i] = np.log2(SigErrors[i] / SigErrors[i+1])



from tabulate import tabulate
print(tabulate([[CellCount[0], UErrors[0], '', SigErrors[0], ''], [CellCount[1], UErrors[1], Rates[0], SigErrors[1], SigRates[0]], [CellCount[2], UErrors[2], Rates[1], SigErrors[2], SigRates[1]], [CellCount[3], UErrors[3], Rates[2], SigErrors[3], SigRates[2]], [CellCount[4], UErrors[4], Rates[3], SigErrors[4], SigRates[3]]], headers = ['Cells' , 'UError', 'URate', 'SigError', 'SigRate']))
