from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="my_model_kernel",
    version="0.0.0",
    ext_modules=[
        CUDAExtension(
            name="my_model_kernel",
            sources = [
                "./my_model_kernel.cpp",
                "./trilinear/src/trilinear_kernel.cu",
                "./trilinear/src/trilinear_cuda.cu",
                "./trilinear/src/trilinear_cpu.cpp",
                "./quadrilinear/src/quadrilinear4d_cuda.cpp",
                "./quadrilinear/src/quadrilinear4d_kernel.cu",
                "./quadrilinear/src/quadrilinear4d.cpp"
            ],
            include_dirs = [ "D:/projects/digitalFilm/models/kernel/trilinear/include",
                             "D:/projects/digitalFilm/models/kernel/quadrilinear/include"],
            extra_compile_args = {
                "cxx": [],
                "nvcc": [ "-O3",
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
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
