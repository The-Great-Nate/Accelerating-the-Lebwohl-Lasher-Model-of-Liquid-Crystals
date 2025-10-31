import sys
import numpy as np
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
#======================================================================
def main(name, size):
    np.savetxt(f"{name}.txt", initdat(size))
    print(f"Saved to {name}.txt!\tHave a nice day!")
    return
#======================================================================
if __name__ == '__main__':
    if int(len(sys.argv)) == 3:
        PROGNAME = sys.argv[0]
        NAME = str(sys.argv[1])
        SIZE = int(sys.argv[2])
        main(NAME, SIZE)
    else:
        print("Usage: python {} <NAME> <SIZE>".format(sys.argv[0]))
#=======================================================================
