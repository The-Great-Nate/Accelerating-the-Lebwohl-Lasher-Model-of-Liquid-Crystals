#!/bin/bash
#!/bin/bash
echo "-------------------------------------------"
echo " CYTHONISE CYTHONISE CYTHONISE "
echo "-------------------------------------------"
python builders/setup_LebwohlLasher_Cython.py build_ext -fi

sizes=(25 50 100 200)

for size in "${sizes[@]}"; do
    echo "============= at $size big. ============="
    for ((i=0; i<20; i++)); do
        echo "-------------------------------------------"
        echo "Running with CYTHON"
        echo "-------------------------------------------"

        python programs/run_LL_Cython.py 500 $size 0.2 0 "Test_Configs/Test_Config_$size.txt"
    done
done
echo "Done with original cythonised code"
