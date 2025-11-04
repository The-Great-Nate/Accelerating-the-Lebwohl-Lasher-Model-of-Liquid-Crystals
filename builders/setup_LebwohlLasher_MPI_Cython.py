#from distutils.core import setup <- Depreciated from Python 3.12 onwards. Thinkpad T480 is on Python 3.13
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from mpi4py import get_include



ext_modules = [
    Extension(
        name="programs.LebwohlLasher_Cython_MPI", 
        sources=["programs/LebwohlLasher_Cython_MPI.pyx"],
        include_dirs=[np.get_include(), get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="programs.LebwohlLasher_Cython_MPI",
    ext_modules=cythonize(ext_modules),
)