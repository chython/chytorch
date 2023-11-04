# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from Cython.Build import build_ext, cythonize
from numpy import get_include
from pathlib import Path
from setuptools import Extension
from setuptools.dist import Distribution
from shutil import copyfile
from sysconfig import get_platform


platform = get_platform()
if platform == 'win-amd64':
    extra_compile_args = ['/O2']
elif platform == 'linux-x86_64':
    extra_compile_args = ['-O3']
else:
    extra_compile_args = []

extensions = [
    Extension('chytorch.utils.data.molecule._unpack',
              ['chytorch/utils/data/molecule/_unpack.pyx'],
              extra_compile_args=extra_compile_args,
              include_dirs=[get_include()]),
]

ext_modules = cythonize(extensions, language_level=3)
cmd = build_ext(Distribution({'ext_modules': ext_modules}))
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    output = Path(output)
    copyfile(output, output.relative_to(cmd.build_lib))
