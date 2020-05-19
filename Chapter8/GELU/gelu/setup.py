from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='gelu',
      ext_modules=[cpp_extension.CUDAExtension('gelu',
                      ['gelu.cc', 'gelu_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
