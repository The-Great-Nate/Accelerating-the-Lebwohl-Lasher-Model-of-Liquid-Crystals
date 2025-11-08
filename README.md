# Lebowhl Lasher Model of Liquid Crystals - Accelerated
<img width="678" height="678" alt="cropped render" src="https://github.com/user-attachments/assets/becfebc2-4add-4cf6-8ecb-cd20d4ceebd5" />


This is a 2D Lebwohl-Lasher liquid crystal model accelerated through various accelerated computing approaches.

The original serial code was modified to allow pre-configured lattices to be input to statistically test the reproduciblity of the parallelised programs.

Otherwise, the code can be located in `programs/LebwohlLasher.py` and was created by Dr. Simon Hanna

## Installation
A `.yml` file is attached to this repository and contains the `mamba` environment used to develop and test all the programs her.

To install: `mamba env create -f environment.yml`

_Feel free to read the `.yml` file for the relevant dependancies_

## Running Programs
### Non-MPI

`python /programs/<PYTHON_FILE> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE>`

### MPI

`mpiexec -n <nproc> python programs/<PYTHON_FILE> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE>`
_where `<nproc>` is the number of workers to use depending on core count on CPU

It is highly recommended that to run any program, especially programs in `builders` and `run_LL_<Acceleration_Method>.py` files on the root directory as all the data is saved in other folders outsie of their directories.

#### _Example Usage:_ Running the Cythoned MPI version of the Code:
1. `python builders/setup_LebwohlLasher_MPI_Cython.py build_ext -fi`
2. `python programs/run_LL_MPI_Cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <FILE>`
3. _Simulation from the inputted configs runs and data is stored in the `data/` folder.

For any desired acceleration method, the file names contain the method used to accelerate the model.

## Notes on Arguments:
| Argument | Description |
| --------- | ----------- |
| ITERATIONS | Number of Monte-Carlo steps to use |
| SIZE | Square Length size of Lattice |
| TEMPERATURE | Reduced Temperature of the System |
| PLOTFLAG | _See table below_ |
| FILE | _OPTIONAL_ Path and Filename of pre-configured Lattice File. It will default to a randomly generated lattice if no file is input. If a number is input with no file extension, it will be treated as a `<THREAD>`.|
| THREADS | _OPTIONAL & Only for `programs/run_LL_Cython_OpenMP.py`_. Number of threads to use. Will default to 1 if no value is input. |

#### Plotflags
| PLOTFLAG | Description |
| --------- | ----------- |
| 0 | no plot of crystal grid |
| 1 | Energy plot of crystal grid |
| 2 | Angles plot of crystal grid |
| 3 | Black plot of crystal grid |
