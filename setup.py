import os
import sys
from setuptools import setup, find_packages

print(
    "Installing. \n Package intended for use with provided conda env. See setup instructions here: https://github.com/liruiw/FleetDrake"
)

if sys.version_info.major != 3:
    print(
        "This Python is only compatible with Python 3, but you are running "
        "Python {}. The installation will likely fail.".format(sys.version_info.major)
    )


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="fleetdrake",
    version="1.0.0",
    packages=find_packages(),
    description="Training and experiment with algorithms for environments in Drake",
    long_description=read("README.md"),
    url="https://github.com/liruiw/FleetDrake.git",
    author="Lirui Wang",
)
