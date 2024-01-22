import setuptools
import versioneer

with open("README.rst", "r") as fh:
    long_description = fh.read()

doc_requirements = [
    "sphinx",
    "sphinx_rtd_theme",
    "ase",
]

full_requirements = [
    "ase",
    "numba",
    "matplotlib",
    "tqdm",
    "graphviz",
    "h5py",
]

setuptools.setup(
    name="hippynn",
    version=versioneer.get_version(),
    author="Nicholas Lubbers et al",
    author_email="hippynn@lanl.gov",
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch",
    ],
    extras_require={"docs": doc_requirements, "full": full_requirements},
    license="BSD 3-Clause License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries",
    ],
    description="The hippynn python package - a modular library for atomistic machine learning with pytorch",
    long_description=long_description,
    packages=setuptools.find_packages(),
    cmdclass=versioneer.get_cmdclass(),
)
