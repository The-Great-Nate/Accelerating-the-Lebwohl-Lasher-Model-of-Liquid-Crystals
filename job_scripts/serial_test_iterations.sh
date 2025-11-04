#!/bin/bash
sizes=(25 50 100 200 500 1000)

for size in "${sizes[@]}"; do
    for ((i=0; i<100; i++)); do
        echo "-------------------------------------------"
        echo "Running with Original Serial"
        echo "-------------------------------------------"

        python programs/LebwohlLasher.py 500 $size 0.2 0 Test_Config_$size.txt
    done
done
echo "Done with original serial code"