import os
import subprocess
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

version_dependent_macros = [
    "-DVERSION_GE_1_1",
    "-DVERSION_GE_1_3",
    "-DVERSION_GE_1_5",
]

extra_cuda_flags = [
    "-std=c++14",
    "-maxrregcount=50",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

cc_number = [75, 80, 86, 89]
cc_flag = []
for n in cc_number:
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{n},code=sm_{n}")

print(cc_flag)

extra_cuda_flags += cc_flag


def get_cuda_bare_metal_version(cuda_dir=CUDA_HOME):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


nvcc_raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version()

setup(
    name="make-torch-inplace",
    version=f"1.4+cuda{bare_metal_major}.{bare_metal_minor}",
    description="PyTorch Tensor matrix operations, inplace!",
    author="Jinmy",
    url="https://git.tianrang-inc.com/imy.jin/inplace-softmax",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "make_torch_inplace": ["csrc/*"],
    },
    ext_modules=[
        CUDAExtension(
            name="make_torch_inplace_C",
            sources=[
                "make_torch_inplace/csrc/pymodule.cpp",
                "make_torch_inplace/csrc/square_matmul.cu",
                "make_torch_inplace/csrc/softmax.cu",
                "make_torch_inplace/csrc/layernorm.cu",
            ],
            include_dirs=[
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "make_torch_inplace/csrc/",
                )
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": (
                    ["-O3", "--use_fast_math"]
                    + version_dependent_macros
                    + extra_cuda_flags
                ),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++"
        "Programming Language :: Python :: 3"
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
        f"Environment :: GPU :: NVIDIA CUDA :: {bare_metal_major}.{bare_metal_minor}",
        "Natural Language :: Chinese (Simplified)",
    ],
    zip_safe=False,
)
