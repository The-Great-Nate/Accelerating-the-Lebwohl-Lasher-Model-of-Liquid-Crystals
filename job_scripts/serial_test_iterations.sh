#!/bin/bash
sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    for ((i=0; i<20; i++)); do
        echo "-------------------------------------------"
        echo "Running with Original Serial"
        echo "-------------------------------------------"

        python programs/LebwohlLasher.py 250 $size 0.2 0 Test_Configs/Test_Config_$size.txt
    done
done
echo "Done with original serial code"
