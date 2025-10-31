#!/bin/bash
echo "-------------------------------------------"
echo " Running program once for compilation"
echo "-------------------------------------------"
python LebwohlLasher_numba.py 100 100 0.2 0 Test_Config.txt


for ((i=0; i<20; i++)); do
    echo "-------------------------------------------"
    echo "Running with Numba JIT compiler"
    echo "-------------------------------------------"

    python LebwohlLasher_numba.py 100 100 0.2 0 Test_Config.txt
done
echo "Done with original serial code"
