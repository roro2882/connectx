from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("gym_connect",["gym_connect.pyx"])]

setup(
        name = "gym_connect app",
        cmdclass= {'build_ext' : build_ext},
        ext_modules=ext_modules
)
