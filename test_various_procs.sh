#!/bin/bash
procs=(1 2 4 6 8 12 14)


for nproc in "${procs[@]}"; do
    echo "-------------------------------------------"
    echo "Running with $nproc MPI process(es)..."
    echo "-------------------------------------------"

    # Run and save output
    mpiexec -n $nproc python LebwohlLasher_MPI.py 100 100 5.6725 0

    echo "Done with $nproc processes."
done