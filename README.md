# fft-stable-fluid
Implementation of a stable fluid solver based on Jos Stam's [Paper](https://dl.acm.org/doi/pdf/10.1145/311535.311548) by Derek Che & Thomas Wang.

This is our final project for APMA 4300, Intro to Numerical Methods. As a project meant for self-learning, we limit the scope to solving a 2-D problem with periodic boundary conditions, where transforming the operation into the Fourier domain gives an elegant and straightforward solution. 

Files:  
`dft`: we built a fast fourier transform package from scratch.  
`project.ipynb`: the main jupyter notebook contains description of the problems, the stable fluid solver, a simulation, and discussions. 
