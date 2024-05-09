from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup


extensions = cythonize(
    [Extension('adeft.score._score', ['src/adeft/score/_score.pyx'])],
    compiler_directives={'language_level': 3},
)

setup(ext_modules=extensions)
