from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension

ext_modules = Extension("SLIM_BPR_Cython_Epoch",
                ["SLIM_BPR_Cython_Epoch.pyx"],
                extra_compile_args=['-O2'],
                include_dirs=[numpy.get_include(),],
                )

setup(
    name="SLIM",
    ext_modules=cythonize(ext_modules),
)