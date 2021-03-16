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
