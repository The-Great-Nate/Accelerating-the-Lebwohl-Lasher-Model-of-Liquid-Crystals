import sys
import LebwohlLasher_Cython_OpenMP as LL_C_OP


if __name__ == '__main__':
    n_args = len(sys.argv)

    if n_args < 4:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> [PLOTFLAG|THREADS] [FILE] [THREADS]")
        sys.exit(1)

    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    print(sys.argv[5])
    # Default values
    FILE = "0"
    THREADS = 1

    # ### 7th argument: only possible if both plotflag and file are given ###
    if n_args >= 7:
        print("I GOT HERE")
        FILE = sys.argv[5]
        THREADS = int(sys.argv[6])

    # ### 6th argument: could be file or plotflag depending on what we already have ###
    elif n_args >= 6:
        if not sys.argv[5].isdigit():
            FILE = sys.argv[5]
        else:
            THREADS = int(sys.argv[5])

    # ### Call main with correct arguments ###
    if FILE == "0":
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, threads = THREADS)
    else:
        print(PLOTFLAG)
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE, THREADS)
#=======================================================================
