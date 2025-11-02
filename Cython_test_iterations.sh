#!/bin/bash
#!/bin/bash
echo "-------------------------------------------"
echo " CYTHONISE CYTHONISE CYTHONISE "
echo "-------------------------------------------"
python setup_LebwohlLasher_Cython.py build_ext -fi


for ((i=0; i<20; i++)); do
    echo "-------------------------------------------"
    echo "Running with CYTHON"
    echo "-------------------------------------------"

    python run_LL_Cython.py 100 100 0.2 0 "Test_Config.txt"
done
echo "Done with original serial code"
