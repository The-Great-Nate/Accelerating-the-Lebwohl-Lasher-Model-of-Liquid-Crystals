#!/bin/bash
threads=(1 2 4)

echo "-------------------------------------------"
echo " CYTHONISE CYTHONISE CYTHONISE "
echo "-------------------------------------------"
python builders/setup_LebwohlLasher_Cython_OpenMP.py build_ext -fi

# This was running on an I5-8250U system with 4 cores in total, modify if needed
sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    for thread in "${threads[@]}"; do
        echo "======================== Running with $threads Threads... ========================"
        for ((i=0; i<20; i++)); do
            echo "----------------------- $i 'th-teration "-----------------------

            # Run and save output
            python programs/run_LL_Cython_OpenMP.py 500 $sizes 0.2 0 Test_Configs/Test_Config_$size.txt $thread

        done
        echo "======================== Done with $threads thread(s). ========================"
    done
done

echo "==========================================="


### Lenovo Thinkpad T480 is the best laptop made. Why is it so upgradable. And WHY IS THE KEYBOARD SO NICE