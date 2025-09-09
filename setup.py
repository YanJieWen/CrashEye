'''
@File: setup.py.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 22, 2025
@HomePage: https://github.com/YanJieWen
'''


import re
import setuptools
import glob
from os import path
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1,3], "Requires PyTorch>=1.3."

# this_dir = path.dirname(path.abspath(__file__))
# with open('README.md','rb') as f:
#     long_description = f.read()

setuptools.setup(
    name='CrashEye',
    version= '1.0.0',
    author= 'Yanjie Wen',
    python_requires = ">=3.6",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)