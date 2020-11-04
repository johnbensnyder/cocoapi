from setuptools import setup, Extension
import numpy as np
import pybind11
import os

os.environ["CC"] = "/opt/amazon/openmpi/bin/mpicxx"
#os.environ["CC"] = "mpicxx"
# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-O3', '-Wno-cpp', '-Wno-unused-function', '-std=c99','-fpermissive'],
    ),
    Extension(
        'ext',
        sources=['pycocotools/ext.cpp'],
        include_dirs = [pybind11.get_include()],
        extra_compile_args=['-O3', '-Wall', '-shared', '-fopenmp', '-std=c++11', '-fPIC'],
        extra_link_args=['-lgomp'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0',
        'pybind11>=2.2',
    ],
    version='2.0+nv0.4.0',
    ext_modules= ext_modules
)
