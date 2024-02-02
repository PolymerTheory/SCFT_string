"# SCFT_string"

Compiling:
use comp.sh (or the line therein) to compile the code

Running:
The code uses CUDA and MPI. It can be run on a single processor but is not setup to run on CPUs. An exam ple of a run line to exwcute the code on 4 processors:
srun --ntasks=4 --gres=gpu:4 ./scft_gpu input.dat

The argument ("input.dat" above) is optional but gives the name of the file from which input parameters are taken.

Input parameters:
Input parameters are arranged as follows (see input.dat for an example):
chiN f Vc
N m[0] m[1] m[2]
D[0] D[1] D[2]
fix1 fix2
flag

Explanation of parameters:
chiN: Flory-Huggisn parameter
f: fractional composition of copolymer, NA/(NA+NB) (NA is the 'tail' length for lipids)
Vc: either the (canonical ensemble) volume fraction of copolymers or (grand canoniocal) fugacity of copolymers
N: number of steps along the copolymer (chain discretization)
m: number of spatial points in the x, y and z directions
D: Length of the box in the x, y and z directions, in units of R0
fix1/2: tells the code if to 'fix' the ends of the string (yes=1)
flag: tells the code if to read W from files or make something up for the initial configuration

W input:
(if flag==1) fields (W) are input from files of the form win_ where _ is the number along the string

There are several hard-coded parts of the code (should be moved to input) such as the number of points along the string (called strn, defined in scft.cu) and whether the system uses the canonical or grand canonical ensemble (see 'props' subroutine in density.h) (default grand canonical).


Outputs:
To screen:
The code outputs parameters etc. to tell you waht stage it has gotten to. In the main loop (solve_field subroutine) it periodically outputs the 'error' (the RMS exchange chemical potential, which tells you how far the field is form equilibrium).

Files:
The code periodically outputs the fields (win_) and A monomer concentration (tail concentration for lipid implementation) (rhoA_temp__.vtk) in a vtk format for ease of visualization. It also outputs a fuke "check" which gives a breakdown of the 'errors'.
After the code has converged (or taken too many steps or been told to exit (see signal handler)) the code calculates the free energy and outputs it to a file 'FEs' together the string position ('alpha'), a breakdown of the free energy and the average lipid composition (only usefulin GC ensemble)

Warning about converfgence:
Short version: Be careful interpreting the error as a simple measure of how converged the system is. Also use other methods like 'run for a while and see if things change'.
Long version: Traditionally, when using SCFT, the error (as defined in the code - root mean squared of the local exchange chemical potential) has been a good way of telling if the system is converged. This is not always the case for complex phases (such as complicated membrane configurations). The 'error' tells you the average 'thermodynamic force' on the system. If the free energy landscape is simple, for example something like a parabola wrt. a simple order parameter, then the system slides towards the minimum, and the 'error', which effectively measures the 'slope' gives a good measure of the 'distance' from the minimum. Consider a system with a set of minima and local maximuma in between. The 'thermodynamic force' near the maxima is small but the system is obviously far from converged. A good way to deal with this is to check the 'direction' of the local exchange chemical potentials (deviations in the solve routine) to see if the system 'wants' to change. Also, just run it for a while after it seems like it converged and see if things are changing. This is not fool-proof, as the system can evolve extremely slowly, so you need to be careful.
