from setuptools import setup, find_packages, Extension
import cluspy
import numpy as np


def _load_requirements():
    with open("requirements.txt", "r") as file:
        requirements = [line.rstrip() for line in file]
    return requirements


dip_extension = Extension('cluspy.utils.dipModule',
                          include_dirs=[np.get_include()],
                          sources=['cluspy/utils/dip.c'])

setup(
    name='ClusPy',
    version=cluspy.__version__,
    packages=find_packages(),
    url='https://github.com/collinleiber/ClusPy',
    license='GPL-3.0 License ',
    author='Collin Leiber',
    author_email='leiber@dbs.ifi.lmu.de',
    description='Clustering in python',
    python_requires='>=3.7',
    install_requires=[_load_requirements()],
    ext_modules=[dip_extension]
)
