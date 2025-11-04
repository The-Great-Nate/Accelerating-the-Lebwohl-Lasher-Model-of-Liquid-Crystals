#!/bin/bash
procs=(1 2 4)
sizes=(25 50 100 200 500 1000)
# This was running on an I5-8250U system with 4 cores in total, modify if needed
for size in "${sizes[@]}"; do
for nproc in "${procs[@]}"; do
    echo "======================== Running with $nproc MPI process(es)... ========================"
    for ((i=0; i<20; i++)); do
        echo "----------------------- $i 'th-teration "-----------------------

        # Run and save output
        mpiexec -n $nproc python programs/LebwohlLasher_MPI.py 500 $size 0.2 0 Test_Config_$size.txt

    done
    echo "======================== Done with $nproc process(es). ========================"
done

echo "==========================================="


### Lenovo Thinkpad T480 is the best laptop made. Why is it so upgradable. And WHY IS THE KEYBOARD SO NICE