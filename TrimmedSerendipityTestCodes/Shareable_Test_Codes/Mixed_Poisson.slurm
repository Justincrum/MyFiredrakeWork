### Slurm script to illustrate how the mixed Poisson problem was solved.
### file: Mixed_Poisson.slurm
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
#SBATCH --job-name=NCF_Poisson_order2
#SBATCH --ntasks=94
#SBATCH --mem=470gb
#SBATCH --time=6:00:00
#SBATCH --partition=standard
#SBATCH --account=ACCOUNT_NAME
#SBATCH --nodes=1
#SBATCH --array 3-3
#SBATCH --output=NCF_3d_Poisson_Order2-N%a-%j.out

module load python/3.8
module swap openmpi3 mpich
###source ~/firedrake/firedrake/bin/activate   #This needs to be activated somehow, based off your path to /firedrake/bin/activate

~/firedrake/firedrake/bin/firedrake-clean

#If not running this slurm script from the directory of the test script, then do a cd command here.
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
mpirun -np 24 python3 Mixed_Poisson_3d.py -O 2 -S ${SLURM_ARRAY_TASK_ID}
