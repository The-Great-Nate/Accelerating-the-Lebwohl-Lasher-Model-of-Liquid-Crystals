import sys
import LebwohlLasher_Cython_OpenMP as LL_C_OP


if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    elif int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        FILE = sys.argv[5]
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE)
    elif int(len(sys.argv)) == 7:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        FILE = sys.argv[5]
        THREADS = int(sys.argv[6])
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
        print("OR WITH 5 ARGS")
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE> <THREADS = 1>".format(sys.argv[0]))
        print("OR WITH 6 ARGS")
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE> <THREADS = THREADS".format(sys.argv[0]))
#=======================================================================
