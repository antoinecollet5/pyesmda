#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

import pyesmda

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

requirements = ["numpy>=1.18.1"]

setup_requirements = ['pytest-runner', 'wheel', ]

test_requirements = ['pytest>=3',]

setup(
    author="Antoine Collet",
    author_email='antoine.collet5@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Ensemble Smoother with Multiple Data Assimilations",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyesmda',
    name='pyesmda',
    packages=find_packages(include=['pyesmda', 'pyesmda.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.com/antoinecollet5/pyesmda',
    version=pyesmda.__version__,
    zip_safe=False,
)
