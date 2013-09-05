#!/usr/bin/env python
'''
Python package for sigmoidal program modeling and optimization
'''

# Copyright 2013 M. Udell
# 
# This file is part of SIGOPT version 0.1.0.
# 
# SIGOPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# SIGOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with SIGOPT.  If not, see <http://www.gnu.org/licenses/>.

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(
    name = "sigopt",
    version = "0.1.1",
    package_dir = {'sigopt': 'lib'},
    packages = ['sigopt','sigopt.test'],
    install_requires = ['docutils>=0.3','numpy','scipy','matplotlib','glpk','cvxopt'],

    # metadata for upload to PyPI
    author='Madeleine Udell',
    author_email='madeleine.udell@gmail.com',
    description='Sigmoidal programming solver',
    url='http://www.stanford.edu/~udell/sigopt',
    keywords = "sigmoid sigmoidal programming optimization non-convex",
    license = "GPL",
    long_description = '''
    Python package for sigmoidal program modeling and optimization
    ''',
    )