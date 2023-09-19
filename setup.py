#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy
from pathlib import Path
from setuptools import setup, Extension, find_namespace_packages


setup(
    name='chytorch',
    version='1.54',
    packages=find_namespace_packages(include=('chytorch.*',)),
    url='https://github.com/chython/chytorch',
    license='LGPLv3',
    author='Dr. Ramil Nugmanov',
    author_email='nougmanoff@protonmail.com',
    python_requires='>=3.8',
    install_requires=['torchtyping>=0.1.4', 'chython>=1.38', 'torch>=1.8', 'scipy>=1.7', 'numpy>=1.20'],
    setup_requires=['cython'],
    ext_modules=[Extension('chytorch.utils.data.molecule._unpack', ['chytorch/utils/data/molecule/_unpack.pyx'],
                           extra_compile_args=['-O3'], include_dirs=[numpy.get_include()])],
    package_data={'chytorch.zoo': ['README.md'], 'chytorch.utils.data.molecule': ['_unpack.pyx']},
    zip_safe=False,
    long_description=(Path(__file__).parent / 'README.md').read_text('utf8'),
    long_description_content_type='text/markdown',
    classifiers=['Environment :: Plugins',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3 :: Only',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Software Development',
                 'Topic :: Software Development :: Libraries',
                 'Topic :: Software Development :: Libraries :: Python Modules']
)
