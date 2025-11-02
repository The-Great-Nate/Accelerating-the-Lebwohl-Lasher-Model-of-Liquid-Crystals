import sys
import LebwohlLasher_MPI as LL_MPI_C


if int(len(sys.argv)) == 5:
    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    LL_MPI_C.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
elif int(len(sys.argv)) == 6:
    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    FILE = sys.argv[5]
    LL_MPI_C.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE)
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
    print("OR WITH 5 ARGS")
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE>".format(sys.argv[0]))