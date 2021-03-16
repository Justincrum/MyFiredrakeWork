#!/bin/bash
#SBATCH --job-name=NCF_Poisson_order2
#SBATCH --ntasks=94
#SBATCH --mem=470gb
#SBATCH --time=6:00:00
#SBATCH --partition=standard
#SBATCH --account=ACCOUNT_NAME
#SBATCH --nodes=1
#SBATCH --array 3-3
#SBATCH --output=NCF_3d_Poisson_Order2-N%a-%j.out

export PATH="/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/bin:/opt/ohpc/pub/compiler/gcc/8.3.0/bin:/opt/ohpc/pub/utils/prun/1.3:/opt/ohpc/pub/utils/autotools/bin:/opt/ohpc/pub/bin:/opt/TurboVNC/bin:/home/u28/jcrum/local/bin:/home/u28/jcrum/firedrake:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
module load python/3.8
module swap openmpi3 mpich
###source ~/firedrake/firedrake/bin/activate   #This needs to be activated somehow, based off the path to /firedrake/bin/activate

~/firedrake/firedrake/bin/firedrake-clean

#If not running this slurm script from the directory of the test script, then do a cd command here.
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
