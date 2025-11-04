#from distutils.core import setup <- Depreciated from Python 3.12 onwards. Thinkpad T480 is on Python 3.13
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name="programs.LebwohlLasher_Cython", 
        sources=["programs/LebwohlLasher_Cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="programs.LebwohlLasher_Cython",
    ext_modules=cythonize(ext_modules),
)