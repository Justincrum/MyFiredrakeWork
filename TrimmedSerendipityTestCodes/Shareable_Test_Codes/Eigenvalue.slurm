### slurm script to give an example of how the eigenvalue problem was solved.
### file: Eigenvalue.slurm
### author:  Justin Crum
### date:  3/19/21

###Copyright <YEAR> <COPYRIGHT HOLDER>

###Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software 
###without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
###permit persons to whom the Software is furnished to do so, subject to the following conditions:

###The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

###THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
###PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
###OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#!/bin/bash
#SBATCH --job-name=Eigenvalue_O2N8
#SBATCH --ntasks=94
#SBATCH --mem=470gb
#SBATCH --time=6:00:00
#SBATCH --partition=standard
#SBATCH --account=ACCOUNT_NAME
#SBATCH --nodes=1
#SBATCH --output=Sminus_Eigenvalues_O2N8-%j.out

module load python/3.8
module swap openmpi3 mpich
source ~/firedrake/firedrake/bin/activate

~/firedrake/firedrake/bin/firedrake-clean
mpirun -np 8 python3 Maxwell_Cavity_3d.py -st_type sinvert -eps_target 3.0 -eps_nev 15 -mat_mumps_icntl_14 200 -mat_mumps_icntl_13 1 -eps_tol 1e-7
mpirun -np 8 python3 Maxwell_Cavity_3d.py -st_type sinvert -eps_target 3.0 -eps_nev 15 -mat_mumps_icntl_14 200 -mat_mumps_icntl_13 1 -eps_tol 1e-7
mpirun -np 8 python3 Maxwell_Cavity_3d.py -st_type sinvert -eps_target 3.0 -eps_nev 15 -mat_mumps_icntl_14 200 -mat_mumps_icntl_13 1 -eps_tol 1e-7
mpirun -np 8 python3 Maxwell_Cavity_3d.py -st_type sinvert -eps_target 3.0 -eps_nev 15 -mat_mumps_icntl_14 200 -mat_mumps_icntl_13 1 -eps_tol 1e-7
