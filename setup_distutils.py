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

from distutils.core import setup

setup(name='sigopt',
      version='0.1.1',
      description='Sigmoidal programming solver',
      author='Madeleine Udell',
      author_email='madeleine.udell@gmail.com',
      url='http://www.stanford.edu/~udell/sigopt',
      package_dir = {'sigopt': 'lib'},
      packages = ['sigopt','sigopt.test'],
      requires = ['numpy','scipy','matplotlib','glpk','cvxopt'],
      long_description = '''
      Python package for sigmoidal program modeling and optimization
      ''',
     )