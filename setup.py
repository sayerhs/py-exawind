# -*- coding: utf-8 -*-

"""\
ExaWind Python Library
======================

Utilities for interacting with computational solvers used in the ExaWind
framework (https://github.com/exawind/)

"""

from setuptools import setup

VERSION = "0.0.1"

classifiers = [
    "Development Status :: 3 -Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Physics"
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Utilities",
]

setup(
    name="exawind",
    version=VERSION,
    url="https://github.com/sayerhs/py-exawind",
    license="Apache License, Version 2.0",
    description="Exawind Python Library",
    long_description=__doc__,
    author="Shreyas Ananthan",
    maintainer="Shreyas Ananthan",
    platforms="any",
    classifiers=classifiers,
    include_package_data=True,
    packages=[
        'exawind',
        'exawind.prelude',
    ],
    entry_points="""
    [console_scripts]
    exawind_nalu=exawind.cli.exawind_nalu:main
    """
)
