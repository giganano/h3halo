
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
	Extension("_fit_driver", ["_fit_driver.pyx", "likelihood.c"])
]

setup(
	name = "src",
	ext_modules = cythonize(extensions)
)

