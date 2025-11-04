#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "programs.LebwohlLasher_Cython_OpenMP",
        ["programs/LebwohlLasher_Cython_OpenMP.pyx"],
        extra_compile_args=['-fopenmp','-v', '-g'],
        include_dirs=[np.get_include()], 
        extra_link_args=['-fopenmp', '-g'],
        #
        #extra_link_args=['-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/13/','-fopenmp'],
    )
]

setup(name="programs.LebwohlLasher_Cython_OpenMP",
      ext_modules=cythonize(ext_modules))

