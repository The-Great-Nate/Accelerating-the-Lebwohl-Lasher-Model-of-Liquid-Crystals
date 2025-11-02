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
from mpi4py import MPI

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
def plotdat(arr,pflag,nmax, leftCol, rightCol):
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
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax, leftCol, rightCol, 0, nmax)
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
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,size,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
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
    filename = "MPI/LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Proc Count        {:d}".format(size),file=FileOut)
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
def one_energy(arr,ix,iy,nmax, leftCol, rightCol, startCol, endCol):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
    nmax (int) = side length of square lattice.
    leftCol (float(nmax)) = array containing column to left of arr;
    rightCol (float(nmax)) = array containing column to right of arr.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
  ########## THIS FUNCTION HAS BEEN MPI'd FOR BEING TOO SLOW >:( ##########
      Each proc handles its own subset of the lattice and the boundary past each proc has to be accounted for
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1) # These are the coordinates
    ixm = (ix-1) # of the neighbours
    iyp = (iy+1)%nmax # with wraparound (MPI funny)
    iym = (iy-1)%nmax #
    
#
# Add together the 4 neighbour contributions
# to the energy
#
    
    local_width = arr.shape[0]
    # Handles case where x co ord is the last col proc can handle,
    if (ix == local_width-1): 
        ang = arr[ix,iy]-rightCol[iy]   # use extra column given to it
    else: 
        ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    # Handles case where x co ord is the first col proc can handle,
    if (ix == 0): 
        ang = arr[ix,iy]-leftCol[iy]   # use extra column given to it
    else: 
        ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    
    # The rest can be handled without using the extra cols
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def all_energy(arr,nmax, comm, rank, leftCol, rightCol, startCol, endCol):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    comm = MPI comm world object
    rank = RANK OF MPI PROCESS
    leftCol (float(nmax)) = array containing column to left of arr;
    rightCol (float(nmax)) = array containing column to right of arr.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
  ########## THIS FUNCTION HAS BEEN MPI'd FOR BEING TOO SLOW >:( ##########    
    Each proc stores its own total energy, but this func needs to return the total energy of lattice
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    enall = 0.0
    enlocal = 0.0
    for i in range(arr.shape[0]):
        for j in range(nmax):
          enlocal += one_energy(arr,i,j,nmax,leftCol,rightCol, startCol, endCol)
    enall = comm.reduce(enlocal, op=MPI.SUM, root = 0) # All proc find energy of their corresponding columns then reduce sum op sends to 0
    
    if rank == 0:
        return enall
    else:
        return 
#=======================================================================
def get_order(arr,nmax, comm, rank):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    comm = MPI comm world object
    rank = RANK OF MPI PROCESS
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
  ########## THIS FUNCTION HAS BEEN MPI'd FOR BEING TOO SLOW >:( ##########    
    Each proc stores its own order paramatrert Qab, but this func needs to combine all Qab results into one
    Reduce method go brrrrr again
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,arr.shape[0],nmax)
    for a in range(3):
        for b in range(3):
            for i in range(arr.shape[0]):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    QabGlobal = comm.reduce(Qab, op = MPI.SUM, root = 0)
    if rank == 0:
      QabGlobal = QabGlobal/(2*nmax*nmax)
      eigenvalues,eigenvectors = np.linalg.eig(QabGlobal)
      return eigenvalues.max()
    else:
      return
#=======================================================================
def MC_step(arr,Ts,nmax, numCols, comm, rank, leftNeighbour, rightNeighbour, leftCol, rightCol, startCol, endCol):
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
    ########## THIS FUNCTION HAS BEEN MPI'd FOR BEING TOO SLOW >:( ##########    
  Generate random indices within what each proc is supposed to handle
  Handle only even then odd columns to prevent conflicts
  Do the MC_step as normal (twice, one for odd and one for even)
  Each proc gets an update on what their neighbouring cols are
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
    xran = np.random.randint(0,high=numCols, size=(numCols,nmax))
    yran = np.random.randint(0,high=nmax, size=(numCols,nmax))
    aran = np.random.normal(scale=scale, size=(numCols,nmax))
    for eo in range(0,2):
      for i in range(eo, numCols,2):
          for j in range(nmax):
              ix = xran[i,j]
              iy = yran[i,j]
              ang = aran[i,j]
              if (ix+iy)%2 != eo: # intention is to look only at odd/even columns
                    continue
              en0 = one_energy(arr,ix,iy,nmax, leftCol, rightCol, startCol, endCol)
              arr[ix,iy] += ang
              en1 = one_energy(arr,ix,iy,nmax, leftCol, rightCol, startCol, endCol)
              if en1<=en0:
                  accept += 1
              else:
              # Now apply the Monte Carlo test - compare
              # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                  boltz = np.exp( -(en1 - en0) / Ts )

                  if boltz >= np.random.uniform(0.0,1.0):
                      accept += 1
                  else:
                      arr[ix,iy] -= ang
      update_boundaries(arr, nmax, comm, rank, leftNeighbour, rightNeighbour, leftCol, rightCol, startCol, endCol)
    acceptGlobal = comm.reduce(accept, op = MPI.SUM, root = 0)
    if rank == 0:
      return acceptGlobal/(nmax*nmax)
    else:
      return None
#=======================================================================
def update_boundaries(arr, nmax, comm, rank, leftNeighbour, rightNeighbour, leftCol, rightCol, startCol, endCol):
  comm.Sendrecv(sendbuf=arr[-1, :], dest=rightNeighbour, sendtag=0, recvbuf=leftCol, source=leftNeighbour, recvtag=0)
  comm.Sendrecv(sendbuf=arr[0, :], dest=leftNeighbour, sendtag=1, recvbuf=rightCol, source=rightNeighbour, recvtag=1)
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
    ####### MPI TYPE THINGS BEGIN #######
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
      # Create and initialise lattice
      if file == 0:
        lattice = initdat(nmax)
      else:
        lattice = np.loadtxt(file)
      # Plot initial frame of lattice
      plotdat(lattice,pflag,nmax, lattice[0,:], lattice[-1,:])
    
    # Create arrays to store energy, acceptance ratio and order parameter 
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    
    numCols = nmax // size # No. cols each proc handles
    startCol = rank * numCols # Index of start col for each proc
    # Gets index of last column each proc handles
    if rank != (size-1): 
        endCol = (rank+1)*numCols
    else:
        endCol = nmax
    
    numCols = endCol - startCol
    cols_per_rank = comm.gather(numCols, root=0)
    
    # Distribute relevant column data to other ranks. Store each chunk in rank 0 first
    if rank == 0:
      chunks = []
      for i in range(size):
          start = i * numCols
          end = (i+1)*numCols if i != size-1 else nmax
          chunks.append(lattice[start:end, :].copy())
    else:
      chunks = None

    # Set up lattice each rank stores locally  in the correct positions
    localLatt = np.zeros((nmax, nmax))
    ownedLatt = comm.scatter(chunks, root=0)  # Rank 0 distributes to other ranks respec. chunks
            
    # Obtain rank of neighbours
    leftNeighbour = (rank-1)%size
    rightNeighbour = (rank+1)%size
    # Store the left and right columns beyond what a rank stores
    leftCol = np.empty(nmax)
    rightCol = np.empty(nmax)

    # Each proc sends boundary cols to next rank
      # Rightmost col is sent to right Neighbour
      # Leftmost col is sent to the left neigbour
    comm.Sendrecv(sendbuf=ownedLatt[-1, :], dest=rightNeighbour, sendtag=0, recvbuf=leftCol, source=leftNeighbour, recvtag=0)
    comm.Sendrecv(sendbuf=ownedLatt[0, :], dest=leftNeighbour, sendtag=1, recvbuf=rightCol, source=rightNeighbour, recvtag=1)
    
    # Set initial values in arrays
    energy[0] = all_energy(ownedLatt,nmax, comm, rank, leftCol, rightCol, startCol, endCol)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(ownedLatt,nmax, comm, rank)
    
    # Begin doing and timing some MC steps.
    initial = MPI.Wtime()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(ownedLatt,temp,nmax, numCols, comm, rank, leftNeighbour, rightNeighbour, leftCol, rightCol, startCol, endCol)
        energy[it] = all_energy(ownedLatt,nmax, comm, rank, leftCol, rightCol, startCol, endCol)
        order[it] = get_order(ownedLatt,nmax, comm, rank)
    final = MPI.Wtime()
    runtime = final-initial

    ### Store final output into Rank 0
    if rank == 0:
      final_lattice = np.zeros((nmax, nmax)) # On the tin
      sendcounts = [] # No. of values each rank will send
      displs = [] # Where each chunk of each rank goes in the final_lattice (ordering by rank is done by default)
      
      for numCols_i in cols_per_rank:
          elements = nmax * numCols_i
          sendcounts.append(elements)
          
      offset = 0
      for count in sendcounts:
          displs.append(offset)
          offset += count
      recvbuf = (final_lattice.ravel(), sendcounts, displs, MPI.DOUBLE) # Final buffer tuple
    else:
      recvbuf = None
    
    owned_flat = ownedLatt.ravel() # Flattens each rank's local lattice to 1D
    comm.Gatherv(sendbuf=owned_flat, recvbuf=recvbuf, root=0) # All ranks write to FInal Lattice
    
    if rank == 0:
      # Final outputs
      print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
      # Plot final frame of lattice and generate output file
      savedat(final_lattice,nsteps,size,temp,runtime,ratio,energy,order,nmax)
      plotdat(final_lattice,pflag,nmax, final_lattice[0,:], final_lattice[-1,:])
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
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        FILE = sys.argv[5]
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
