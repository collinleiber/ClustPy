from setuptools import setup, Extension
import numpy as np


dip_extension = Extension('clustpy.utils.dipModule',
                          include_dirs=[np.get_include()],
                          sources=['clustpy/utils/dip.c'])


if __name__ == "__main__":
    setup(ext_modules=[dip_extension])
