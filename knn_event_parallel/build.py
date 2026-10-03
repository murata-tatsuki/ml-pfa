"""Build only this optional extension, without installing/overwriting torch_cmspepr.

Run from the repository root, using the same Python/CUDA environment as training:
TORCH_CUDA_ARCH_LIST=9.0 python knn_event_parallel/build.py build_ext --inplace
"""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent.parent
setup(
    name='pfa-knn-event-parallel',
    packages=['knn_event_parallel'],
    package_dir={'knn_event_parallel': str(ROOT / 'knn_event_parallel')},
    ext_modules=[CUDAExtension(
        'knn_event_parallel._C',
        [str(ROOT / 'knn_event_parallel/csrc' / name) for name in
         ('select_knn_cuda.cpp', 'select_knn_cuda_kernel.cu')],
        # Match legacy flags: in particular, do not enable fast math.
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['--expt-relaxed-constexpr', '-O2']},
        extra_link_args=['-s'],
    )],
    cmdclass={'build_ext': BuildExtension.with_options(
        no_python_abi_suffix=True, use_ninja=False)},
)
