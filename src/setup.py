#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import setuptools
from Cython.Build import cythonize
from Cython.Distutils import build_ext

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="midr",
    version="1.2.8",
    packages=['midr'],
     setup_requires=[
         'scipy>=1.3',
         'numpy>=1.16',
         'cython>=0.28.0'
    ],
    install_requires=[
        'scipy>=1.3',
        'numpy>=1.16',
        'pynverse>=0.1',
        'pandas>=0.25.0',
        'mpmath>=1.1.0',
        'cython>=0.28.0'
    ],
    author="Laurent Modolo",
    author_email="laurent.modolo@ens-lyon.fr",
    description="Compute idr from m NarrowPeak files and a merged NarrowPeak",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitbio.ens-lyon.fr/LBMC/sbdm/midr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, \
version 2.1 (CeCILL-2.1)",
        "Operating System :: OS Independent"
    ],
    test_suite='pytest',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['midr=midr.__main__:main'],
    },
    cmdclass = {'build_ext': CustomBuildExtCommand},
    ext_modules=cythonize(
        module_list=["midr/*.pyx"],
        language_level=3
    ),
    package_data={
        'midr': ['*.pyx']
    },
    zip_safe=False
)
