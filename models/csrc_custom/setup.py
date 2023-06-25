import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
# "helper_math.h" is copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='vren_custom',
    version='0.1',
    author='jimliu',
    author_email='pingfan_zhilu_law268@sjtu.edu.cn',
    description='modified cuda volume rendering library',
    long_description='modified cuda volume rendering library for custom use',
    ext_modules=[
        CUDAExtension(
            name='vren_custom',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)