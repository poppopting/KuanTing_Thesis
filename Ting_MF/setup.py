#!/usr/bin/env python
# coding: utf-8

# In[2]:


from distutils.core import setup
from Cython.Build import cythonize


from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


#sourcefiles = ['ting_svd.pyx', 'ting_svdpp.pyx']

#extensions = [Extension("example", sourcefiles)]


#setup(
#    ext_modules = cythonize(extensions),
#    include_dirs=[np.get_include()]
#)


setup(
    ext_modules = cythonize(["*.pyx"]),
    include_dirs=[np.get_include()]
)