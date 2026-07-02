from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Define as flags de acordo com o sistema operacional
if sys.platform.startswith("win"):
    openmp_arg = ["/openmp"]
else:
    openmp_arg = ["-fopenmp"]

extensions = [
    Extension(
        "skeleton",
        sources=["skeleton.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
        extra_compile_args=openmp_arg,  # Força a flag correta no Linux
        extra_link_args=openmp_arg      # Força a flag correta no Linux
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)