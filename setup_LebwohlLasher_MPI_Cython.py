#from distutils.core import setup <- Depreciated from Python 3.12 onwards. Thinkpad T480 is on Python 3.13
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from mpi4py import get_include

setup(name="LebwohlLasher_MPI_Cython",
      ext_modules=cythonize("LebwohlLasher_MPI.pyx"),
      include_dirs=[np.get_include(), get_include()], #ensures module loaded for compilation
      extra_compile_args=["-O3"]
)