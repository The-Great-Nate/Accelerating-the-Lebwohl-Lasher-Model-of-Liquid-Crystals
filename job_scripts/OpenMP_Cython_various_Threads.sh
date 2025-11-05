#!/bin/bash

# This was running on an Ryzen 7 9800X3D system with 16 Threads in total, modify if needed
threads=(1 2 4 6 8 12 16)

echo "-------------------------------------------"
echo " CYTHONISE CYTHONISE CYTHONISE "
echo "-------------------------------------------"
python builders/setup_LebwohlLasher_Cython_OpenMP.py build_ext -fi


sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    for thread in "${threads[@]}"; do
        echo "======================== Running with $thread Threads of size $size... ========================"
        for ((i=0; i<20; i++)); do
            echo "----------------------- $i 'th-teration "-----------------------

            # Run and save output
            python programs/run_LL_Cython_OpenMP.py 250 $size 0.2 0 Test_Configs/Test_Config_$size.txt $thread

        done
        echo "======================== Done with $thread thread(s). ========================"
    done
done

echo "==========================================="


### Lenovo Thinkpad T480 is the best laptop made. Why is it so upgradable. And WHY IS THE KEYBOARD SO NICE