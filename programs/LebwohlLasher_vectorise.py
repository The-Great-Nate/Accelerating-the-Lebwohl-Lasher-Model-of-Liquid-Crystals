"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    arr2d = arr.reshape((nmax,nmax))
    u = np.cos(arr2d)
    v = np.sin(arr2d)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros(nmax * nmax)
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                flattened_index = i*nmax +j
                cols[flattened_index] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    cols = cols.reshape((nmax*nmax))
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data; <---- It's not even used here so I deleted it :)
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "data/Vectorised/LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #

    flattened_index = ix*nmax + iy
    flattened_index_xp = ixp*nmax + iy
    flattened_index_xm = ixm*nmax + iy
    flattened_index_yp = ix*nmax + iyp
    flattened_index_ym = ix*nmax + iym
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[flattened_index]-arr[flattened_index_xp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[flattened_index]-arr[flattened_index_xm]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[flattened_index]-arr[flattened_index_yp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[flattened_index]-arr[flattened_index_ym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
#=======================================================================
def all_energy(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """

    arr = arr.reshape((nmax,nmax)) # Needed for the below function's indexing to work correctly

    ### Arrays to store each direction's energy contribution.
      # Making empty arrays as temporary storage for each direction
    energyArr = np.full_like(arr,0)
    ang = np.empty_like(arr)
    shift = np.empty_like(arr)

    #left comparison
    shift[:,:-1] = arr[:,1:]
    shift[:,-1] = arr[:,0]
    ang = arr-shift
    energyArr = energyArr + 0.5*(1.0 - 3.0*np.cos(ang)**2)

    #right comparison
    shift[:,1:] = arr[:,:-1]
    shift[:,0] = arr[:,-1]
    ang = arr-shift
    energyArr = energyArr + 0.5*(1.0 - 3.0*np.cos(ang)**2)

    #down comparison
    shift[1:][:] = arr[:-1][:]
    shift[0][:] = arr[-1][:]
    ang = arr-shift
    energyArr = energyArr + 0.5*(1.0 - 3.0*np.cos(ang)**2)

    #up comparison
    shift[:-1][:] = arr[1:][:]
    shift[-1][:] = arr[0][:]
    ang = arr-shift
    energyArr = energyArr + 0.5*(1.0 - 3.0*np.cos(ang)**2)

    enall = np.sum(energyArr)
    
    return enall
#=======================================================================
def get_order(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    nmax_squared = nmax * nmax
    #
    # Attempted to replicate the functionality of get_order
    # based on looking at how a and b changed in the loop in comparison to accessing Qab
    #

    cos_stack = np.cos(arr)
    sin_stack = np.sin(arr)
    cosSin = np.sum(3 * cos_stack * sin_stack) #first term in Qab[a,b] term
    cosSq = np.sum(3*np.power(np.cos(arr),2)) - nmax_squared
    sinSq = np.sum(3*np.power(np.sin(arr),2)) - nmax_squared
    
    Qab[0,0] = cosSq
    Qab[0,1] = cosSin
    Qab[1,0] = cosSin
    Qab[1,1] = sinSq
    Qab[2,2] = -nmax_squared

    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()

#=======================================================================
def MC_step(arr,Ts,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    scale=0.1+Ts
    accept = 0
    nmax_squared = nmax * nmax
    xran = np.random.randint(0,high=nmax, size=nmax_squared)
    yran = np.random.randint(0,high=nmax, size=nmax_squared)
    aran = np.random.normal(scale=scale, size=nmax_squared)

    ### Sort by row then by column for sequential sampling
    sort_idx = np.lexsort((yran, xran)) 

    xran = xran[sort_idx]
    yran = yran[sort_idx]
    aran = aran[sort_idx]
    
    ### "The for loop is the worst invention known to mankind.."
    ### I couldn't get rid of this one when trying to increase access efficiency soz :'(
    for i in range(nmax_squared):
          ix = xran[i]
          iy = yran[i]
          ang = aran[i]
          en0 = one_energy(arr,ix,iy,nmax)
          flattened_index = ix*nmax + iy
          arr[flattened_index] += ang
          en1 = one_energy(arr,ix,iy,nmax)
          if en1<=en0:
              accept += 1
          else:
          # Now apply the Monte Carlo test - compare
          # exp( -(E_new - E_old) / T* ) >= rand(0,1)
              boltz = np.exp( -(en1 - en0) / Ts )

              if boltz >= np.random.uniform(0.0,1.0):
                  accept += 1
              else:
                  arr[flattened_index] -= ang
    return accept/(nmax*nmax)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, file = 0):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    if file == 0:
        lattice = initdat(nmax)
    else:
        lattice = np.loadtxt(file)

    arr = lattice.ravel()
    # Plot initial frame of lattice
    plotdat(arr,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)
    
    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(arr,temp,nmax)
        energy[it] = all_energy(arr,nmax)
        order[it] = get_order(arr,nmax)
    final = time.time()
    runtime = final-initial

    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(arr,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    elif int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        FILE = sys.argv[5]
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
        print("OR WITH 5 ARGS")
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE>".format(sys.argv[0]))
#=======================================================================

