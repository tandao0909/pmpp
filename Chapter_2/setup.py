import glob
import shutil
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def get_extensions():
    # CUDA detection:
    # - torch.cuda.is_available() can be True even when nvcc isn't usable for building extensions.
    # - torch's CUDA_HOME is derived from $CUDA_HOME; in your env it's set to /usr/local/cuda-12.8,
    #   but that path may not exist / contain nvcc (common on systems with distro nvcc in /usr/bin).
    cuda_home = CUDA_HOME
    nvcc_in_path = shutil.which("nvcc")
    nvcc_at_cuda_home = (
        cuda_home is not None and (Path(cuda_home) / "bin" / "nvcc").exists()
    )

    # Only build CUDA sources when nvcc is available at CUDA_HOME.
    # Otherwise fall back to a pure C++ extension (still works with torch headers / ATen ops).
    use_cuda = bool(torch.cuda.is_available() and nvcc_at_cuda_home)
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": ["-O3"],
    }

    current_dir = Path(__file__).parent
    extension_dir = current_dir / "extension_cpp" / "csrc"
    # Always compile C++ sources.
    sources = list(glob.glob((extension_dir / "*.cpp").as_posix()))

    # Only compile CUDA sources when CUDA is available.
    if use_cuda:
        extension_cuda_dir = extension_dir / "cu"
        cuda_sources = list(glob.glob((extension_cuda_dir / "*.cu").as_posix()))
        sources += cuda_sources

    ext_modules = [
        extension(
            "extension_cpp._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=True,
        )
    ]
    return ext_modules


setup(
    name="extension_cpp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Extension for the project",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
