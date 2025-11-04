#!/bin/bash
sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    for ((i=0; i<20; i++)); do
        echo "-------------------------------------------"
        echo "Running Vectorised Form"
        echo "-------------------------------------------"

        python programs/LebwohlLasher_vectorise.py 500 $size 0.2 0 Test_Configs/Test_Config_$size.txt
    done
done
echo "Done with Vectorised code code"