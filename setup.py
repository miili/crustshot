#!/bin/python
from setuptools import setup

setup(
    name='crustshot',
    version='0.0.1',
    description='Query USGS Crustal database',
    author='Marius P. Isken',
    author_email='marius.isken@gfz-potsdam.de',
    license='GPL',
    install_requires=['numpy>=1.9.0', 'matplotlib'],
    packages=['crustshot'],
    package_dir={'crustshot': 'src'},
    data_files=[('crustshot/data', ['data/gsc20130501.txt'])])
