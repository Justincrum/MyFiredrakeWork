Trimmed serendipity example codes

**Authors:**  

Justin Crum, Andrew Gillette, Josh Levine, Rob Kirby

**Email:** 

jcrum@email.arizona.edu, agillette@math.arizona.edu,
josh@email.arizona.edu, robert_kirby@baylor.edu


This includes sample codes used to run numerical experiments comparing tensor
product and trimmed serendipity elements.  It also includes a few sample SLURM
scripts to give an idea of how to run and reproduce results in a HPC setting.

**Prerequisites:**

To run the following test python scripts, you need to have Firedrake installed 
and the `TrimmedSerendipity` branch checked out in the following directories:

- `fiat`
- `FInAT`
- `ufl`
- `tsfc`

**Files (elements that can be used in them listed in paranthesis):**

- Projection_2d.py  (RTCF, RTCE, Lagrange, SminusDiv, SminusCurl, S)
- Projection_3d.py  (NCF, NCE, Lagrange, SminusDiv, SminusCurl, S)
- Primal_Poisson_3d.py  (Lagrange, S)
- Mixed_Poisson_3d.py   (NCF and DQ, SminusDiv and DPC)
- Maxwell_Cavity_3d.py  (NCE, SminusCurl)
- Mixed_Poisson.slurm
- Eigenvalue.slurm

**Directions:**

To run any of the python scripts *except* for `Maxwell_Cavity_3d.py`, you may run
the command `python ____.py -O # -N ##` where the number after `O` will 
determine the the order of the element used, and the number after `N` will
determine the exponent for the mesh of size 2^N x 2^N or 2^N x 2^N x 2^N.  
Note that both O and N should be positive integers.  

To change these parameters in `Maxwell_Cavity_3d.py`, please edit the 
script file itself.  It is possible to add on command line parameters to this 
code, for examples of this see Eigenvalue.slurm or the SLEPc manual.  To run
this code without extra command line parameters, 
do `python Maxwell_Cavity_3d.py`.

**Results:**

- The codes `Projection_2d.py` and `Projection_3d.py` were used to create the
 results in figure 4 of the paper.
- The codes `Primal_Poisson_2d.py`, `Mixed_Poisson_2d.py`, `Primal_Poisson_3d.py`,
 and `Mixed_Poisson_3d.py` were used to create results in both figures 5 and 6.
- The code `Maxwell_Cavity_3d.py` was used to create table 3 and the
 results in figure 7.
