from distutils.core import setup, Extension
from distutils import sysconfig
import numpy.distutils.misc_util
import os, glob

print(sysconfig.get_config_vars()['CFLAGS'])

cpp_files = []
cpp_files.append("ailib.cpp")

#compile deeplearning folder
for file in os.listdir("../deeplearning/"):
    if file == "Dataset.cpp": continue
    if file.endswith(".cpp"): cpp_files.append(os.path.join("../deeplearning/", file))

#compile util folder
for file in os.listdir("../util/"):
    if file.endswith(".cpp"):
        cpp_files.append(os.path.join("../util/", file))

#compile reinforcementlearning folder
for file in os.listdir("../RL/"):
    if file.endswith(".cpp"):
        cpp_files.append(os.path.join("../RL/", file))

#compile visualization folder
"""
for file in os.listdir("../visualization/"):
    if file.endswith(".cpp"):
        cpp_files.append(os.path.join("../visualization/", file))
"""

c_ext = Extension("_ailib", cpp_files, extra_compile_args=['-std=c++11'], language="c++")

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
