#!/bin/bash
#!/bin/bash
echo "-------------------------------------------"
echo " CYTHONISE CYTHONISE CYTHONISE "
echo "-------------------------------------------"
python builders/setup_LebwohlLasher_MPI_Cython.py build_ext -fi

procs=(1 2 4 6 8)
sizes=(25 50 100 200)
# This was running on an Ryzen 7 9800X3D system with 8 cores in total, modify if needed
for size in "${sizes[@]}"; do
    for nproc in "${procs[@]}"; do
        echo "======================== Running with $nproc MPI process(es)... (cython) at $size big. ========================"
        for ((i=0; i<20; i++)); do
            echo "----------------------- $i 'th-teration "-----------------------

            # Run and save output
            mpiexec -n $nproc python programs/run_LL_MPI_Cython.py 250 $size 0.2 0 Test_Configs/Test_Config_$size.txt

        done
        echo "======================== Done with $nproc process(es). ========================"
    done
done
echo "==========================================="
