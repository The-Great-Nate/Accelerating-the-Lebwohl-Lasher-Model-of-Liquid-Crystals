#!/bin/bash
procs=(1 2 4 6 8 12 14)

# This was running on an I9-12900H system with 14 cores in total, modify if needed
for nproc in "${procs[@]}"; do
    echo "-------------------------------------------"
    echo "Running with $nproc MPI process(es)..."
    echo "-------------------------------------------"

    # Run and save output
    mpiexec -n $nproc python LebwohlLasher_MPI.py 100 100 5.6725 0

    echo "Done with $nproc processes."
done