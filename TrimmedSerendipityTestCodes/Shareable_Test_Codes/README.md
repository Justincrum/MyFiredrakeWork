Trimmed serendipity example codes

**Author:**  Justin Crum  **Email:** jcrum@email.arizona.edu

This includes sample codes used to run numerical experiments comparing tensor product and trimmed serendipity elements.
It also includes a few sample SLURM scripts to give an idea of how to run and reproduce results in a HPC setting.

**Files:**

- Projection_2d.py
- Projection_3d.py
- Primal_Poisson_3d.py
- Mixed_Poisson_3d.py
- Maxwell_Cavity_3d.py
- Mixed_Poisson.slurm
- Eigenvalue.slurm

**Directions:**
To run any of the python scripts *except* for Maxwell_Cavity_3d.py, you may run the command 
`python ____.py -O # -S ##`

where the number after `O` will determine the the order of the element used, and the number after `S` will
determine the size of the mesh.  To change these parameters in Maxwell_Cavity_3d.py, please edit the script file itself.

To change finite elements used in any of the python scripts, the `FunctionSpace` line(s) must be changed.  The following
elements can all be used in certain scenarios:

- NCE
- RTCE
- NCF
- RTCF
- Lagrange
- SminusDiv
- SminusCurl
- S
