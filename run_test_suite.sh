#!/bin/bash

./job_scripts/serial_test_iterations.sh
./job_scripts/vectorise_test_iterations.sh
./job_scripts/numba_test_iterations.sh
./job_scripts/MPI_test_various_procs.sh
./job_scripts/Cython_test_iterations.sh
./job_scripts/MPI_Cython_test_iterations.sh
./job_scripts/OpenMP_Cython_various_Threads.sh
