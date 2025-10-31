#!/bin/bash
procs=(1 2 4 6 8)

# This was running on an Ryzen 7 9800X3D system with 14 cores in total, modify if needed
for nproc in "${procs[@]}"; do
    echo "======================== Running with $nproc MPI process(es)... ========================"
    for ((i=0; i<20; i++)); do
        echo "----------------------- $i 'th-teration "-----------------------

        # Run and save output
        mpiexec -n $nproc python LebwohlLasher_MPI.py 100 100 0.2 0 Test_Config.txt

    done
    echo "======================== Done with $nproc process(es). ========================"
done

echo "==========================================="

for ((i=0; i<100; i++)); do
    echo "-------------------------------------------"
    echo "Running with Original Serial"
    echo "-------------------------------------------"

    python LebwohlLasher.py 100 100 0.2 0 Test_Config.txt
done
echo "Done with original serial code"
