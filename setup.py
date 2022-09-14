import os.path

from setuptools import setup


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, "doe/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return ""


root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="basf-doe",
    author="BASF",
    license="BSD-3",
    url="https://github.com/basf/doe",
    keywords=[
        "Design of experiments",
        "Experimental design",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=get_version(),
    description="Advanced & flexible design of experiments",
    packages=["doe"],
    python_requires=">=3.7",
    install_requires=[
        "formulaic>=0.5",
        "loguru",
        "mopti",
        "numpy",
        "pandas",
        "scipy",
    ],
    extras_require={
        "tests": ["pytest"],
        "docs": [
            "mkdocs==1.3.0",
            "mkdocs-material==8.2.1",
            "mkdocstrings==0.19.0",
            "mkdocstrings-python-legacy==0.2.3",
        ],
    },
)
