#!/bin/bash
for ((i=0; i<100; i++)); do
    echo "-------------------------------------------"
    echo "Running with Original Serial"
    echo "-------------------------------------------"

    python LebwohlLasher.py 100 100 0.2 0 Test_Config.txt
done
echo "Done with original serial code"