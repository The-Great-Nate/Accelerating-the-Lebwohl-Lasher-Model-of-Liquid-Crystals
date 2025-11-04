#!/bin/bash
echo "-------------------------------------------"
echo " Running program once for compilation"
echo "-------------------------------------------"
python programs/LebwohlLasher_numba.py 3 25 0.2 0 Test_Configs/Test_Config_25.txt

sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    for ((i=0; i<20; i++)); do
        echo "-------------------------------------------"
        echo "Running with Numba JIT compiler"
        echo "-------------------------------------------"

        python programs/LebwohlLasher_numba.py 500 $size 0.2 0 Test_Configs/Test_Config_$size.txt
    done
done
echo "Done with numba serial code"
