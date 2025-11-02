#from distutils.core import setup <- Depreciated from Python 3.12 onwards. Thinkpad T480 is on Python 3.13
from setuptools import setup
from Cython.Build import cythonize

setup(name="LebwohlLasher_Cython",
      ext_modules=cythonize("LebwohlLasher_Cython.pyx"))

