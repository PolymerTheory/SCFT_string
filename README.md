# SCFT_string

## Compiling
Use `comp.sh` (or the line therein) to compile the code.

## Running
The code uses CUDA and MPI. It can be run on a single processor but is not setup to run on CPUs. Example of a run line to execute the code on 4 processors:
srun --ntasks=4 --gres=gpu:4 ./scft_gpu input.dat

The argument ("input.dat" above) is optional but specifies the file from which input parameters are taken.

## Explanation of Parameters
- **chiN:** Flory-Huggisn parameter
- **f:** fractional composition of copolymer, NA/(NA+NB) (NA is the 'tail' length for lipids)
- **phidb:** either the (canonical ensemble) volume fraction of copolymers or (grand canonical) fugacity of copolymers
- **ens:** statistical ensemble: 1 = canonical, 2 = grand canonical
- **N:** number of steps along the copolymer (chain discretization)
- **m:** number of spatial points in the x, y, and z directions
- **D:** Length of the box in the x, y, and z directions, in units of R0
- **fix:** tells the code if to 'fix' the ends of the string. Takes 2 entires: first is for the first replica and second is the the last
- **flag:** tells the code if to read W from files or make something up for the initial configuration
- **justFE:** tells the code if to just calculate free energy (i.e., leave W_- as is)
- **dostring:** tells the code if to use the string method i.e., couple the replicas

The parameters that look like they should be boolians (e.g. 'dostring') take integers with 0=no, 1=yes

## Outputs
### To Screen
The code outputs parameters etc., to inform you about the current stage. It periodically outputs the 'error' (the RMS exchange chemical potential) in the main loop (solve_field subroutine) to indicate how far the field is from equilibrium.

### Files
The code periodically outputs fields (win_) and A monomer concentration (rhoA__.vtk) in a VTK format for ease of visualization. It also outputs a "check" file detailing 'errors'. After convergence, it calculates and outputs the free energy to a file 'FEs' together with the string position ('alpha'), the free energy, and the average lipid composition.

## Warning about Convergence
###Short version:  
Be cautious interpreting the error as a measure of convergence. Use additional methods like running for an extended period to check for changes. The 'error' might not always accurately represent how close the system is to convergence, especially in complex phases.  

###Long version:  
Traditionally, when using SCFT, the error (as defined in the code - root mean squared of the local exchange chemical potential) has been a good way of telling if the system is converged. This is not always the case for complex phases (such as complicated membrane configurations). The 'error' tells you the average 'thermodynamic force' on the system. If the free energy landscape is simple, for example something like a parabola wrt. a simple order parameter, then the system slides towards the minimum, and the 'error', which effectively measures the 'slope' gives a good measure of the 'distance' from the minimum. Consider a system with a set of minima and local maximuma in between. The 'thermodynamic force' near the maxima is small but the system is obviously far from converged. A good way to deal with this is to check the 'direction' of the local exchange chemical potentials (deviations in the solve routine) to see if the system 'wants' to change. Also, just run it for a while after it seems like it converged and see if things are changing. This is not fool-proof, as the system can evolve extremely slowly, so you need to be careful. Considerations like this are particularly important when studying membranes: the free energy landscape can be pretty flat and thus seem converged, but carefully inspecting may reveal that a membrane wants to, for example, translate normal to its plane. 

