from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import torch
import os
kernel_dir = os.path.dirname(os.path.abspath(__file__))

cuda_available = torch.cuda.is_available()

common_sources = [
    "./my_model_kernel.cpp",
    "./trilinear/src/trilinear_cpu.cpp",
    "./quadrilinear/src/quadrilinear4d.cpp",
]

cuda_sources = [
    "./trilinear/src/trilinear_kernel.cu",
    "./trilinear/src/trilinear_cuda.cu",
    "./quadrilinear/src/quadrilinear4d_cuda.cpp",
    "./quadrilinear/src/quadrilinear4d_kernel.cu",
]

include_dirs = [
    os.path.join(kernel_dir, "trilinear", "include"),
    os.path.join(kernel_dir, "quadrilinear", "include"),
]

if cuda_available:
    ext = CUDAExtension(
        name="my_model_kernel",
        sources=common_sources + cuda_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": ["-DWITH_CUDA"],
            "nvcc": ["-O3",
                     "-std=c++17",
                     "-Wno-deprecated-gpu-targets",
                     "-U__CUDA_NO_HALF_OPERATORS__",
                     "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extended-lambda",
                     "--use_fast_math",
                     "--ptxas-options=-v",
                     "-lineinfo",
                     "--threads", "16", "-allow-unsupported-compiler"],
        }
    )
else:
    ext = CppExtension(
        name="my_model_kernel",
        sources=common_sources,
        include_dirs=include_dirs,
        extra_compile_args=["-fopenmp"],
    )

setup(
    name="my_model_kernel",
    version="0.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
