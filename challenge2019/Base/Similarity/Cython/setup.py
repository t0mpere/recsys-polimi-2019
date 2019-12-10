from distutils.core import setup

import numpy
from Cython.Build import cythonize
from setuptools import Extension

ext_modules = Extension("Compute_Similarity_Cython",
                ["Compute_Similarity_Cython.pyx"],
                extra_compile_args=['-O2'],
                include_dirs=[numpy.get_include(),],
                )

setup(
    name="SLIM",
    ext_modules=cythonize(ext_modules),
)