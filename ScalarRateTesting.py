#Creating a file to understand how the brezzi douglas marini finite element code works.
#Justin Crum


from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
PolyDegree = 3
Errors = []
CellCount = []
Times = 5


#Now we want to try solving the pde as in RateTesting.py.


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
    #plot(mesh)
    #plt.show()

    #Set up the function space
    V = FunctionSpace(mesh, "S", PolyDegree)

    #Trial and Test functions.
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    #Setting up the pde.
    f = Function(V) #RHS
    a = (dot(grad(u), grad(v))) * dx #LHS
    L = f * v * dx
    bc = DirichletBC(V, project(sin(x)*(e**y), V), "on_boundary")
    w = Function(V) #Solution initialized

    params = {"snes_type": "newtonls",

          "snes_linesearch_type": "basic",

          "snes_lag_jacobian": -2,

          "snes_lag_preconditioner": -2,

          "ksp_type": "preonly",

          "snes_max_it": 2,

          "pc_type": "lu",

          "snes_rtol": 1e-16,

          "snes_atol": 1e-25}

    #solver_params = {'ksp_atol' : 1e-30, 'ksp_rtol' : 1e-16, 'ksp_divtol' : 1e7}
    solve(a == L, w, bcs = [bc], solver_parameters = params)# solver_params)


    #Create the exact solution and compute the error value.
    Exact = project(sin(x)*(e**y), V)
    #This line computes the error norm between the two functions.
    ErrVal = norms.errornorm(Exact, w, "L2", degree_rise = None, mesh = None)

    #This line computes the error norm between the two gradients.
    #ErrVal = norms.norm(grad(Exact) - grad(w), norm_type = 'L2', mesh=None)
    
    Errors.append(ErrVal)
    CellCount.append(Cells)




#import matplotlib.pyplot as plt
#plot(w)
#plt.show()

#Determining the convergence rate.
Errors = np.array(Errors)
Leng = np.max(np.shape(Errors))
Rates = np.zeros([Leng - 1, 1])
for i in range(0, Times):
    h1 = 1.0 / CellCount[i]
    h2 = 1.0 / CellCount[i+1]
    Rates[i] = np.log2(Errors[i] / Errors[i+1])

    
#print(Rates)
#print(Errors)
from tabulate import tabulate
print(tabulate([[CellCount[0], 25, Errors[0], ''], [CellCount[1], 81, Errors[1], Rates[0]], [CellCount[2], 289, Errors[2], Rates[1]], [CellCount[3], 1089, Errors[3], Rates[2]], [CellCount[4], 4225, Errors[4], Rates[3]]], headers = ['Cells' , 'DoFs', 'Error', 'Rate']))
