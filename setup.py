#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pkg_resources
import sys

from setuptools import find_packages
from setuptools import setup


setup_requires = []
install_requires = [
    'opencv-python',
    'numpy',
    'matplotlib',
    'tensorflow',                    
    'tqdm',
]


setup(
    name='tf2_transformers',
    version='0.0.0',
    description='',
    author='Krishneel Chaudhary',
    author_email='krishneel@krishneel',
    license='None',
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires
)
