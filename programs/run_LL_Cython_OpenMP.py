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

    # Default values
    PLOTFLAG = 0
    FILE = None
    THREADS = 1

    ### Determine meaning of the 4th argument (after temperature) ###
    if n_args >= 5:
        arg4 = sys.argv[4]
        if arg4.isdigit():
            THREADS = int(arg4)        # Treat as THREADS
        else:
            PLOTFLAG = int(arg4)       # Treat as PLOTFLAG

    # ### 6th argument: could be file or plotflag depending on what we already have ###
    if n_args >= 6:
        if FILE is None and not sys.argv[5].isdigit():
            FILE = sys.argv[5]
        elif PLOTFLAG == 0 and sys.argv[5].isdigit():
            PLOTFLAG = int(sys.argv[5])
        elif THREADS == 1 and sys.argv[5].isdigit():
            THREADS = int(sys.argv[5])

    # ### 7th argument: only possible if both plotflag and file are given ###
    if n_args >= 7:
        if FILE is None and not sys.argv[6].isdigit():
            FILE = sys.argv[6]
        elif THREADS == 1 and sys.argv[6].isdigit():
            THREADS = int(sys.argv[6])

    # ### Call main with correct arguments ###
    if FILE is None:
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        LL_C_OP.main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, FILE, THREADS)
#=======================================================================
