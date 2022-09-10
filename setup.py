import os.path

from setuptools import setup


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, "doe/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return ""


setup(
    name="basf-doe",
    version=get_version(),
    description="Advanced & flexible design of experiments",
    packages=["doe"],
    install_requires=[
        "formulaic<0.5",
        "loguru",
        "mopti",
        "numpy",
        "pandas",
        "scipy",
    ],
    python_requires=">=3.6",
)
