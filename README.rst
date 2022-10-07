=======
pyESMDA
=======


.. image:: https://img.shields.io/badge/License-MIT license-blue.svg
    :target: https://gitlab.com/antoinecollet5/pyesmda/-/blob/master/LICENSE

.. image:: https://img.shields.io/badge/dynamic/json?color=green&label=gitlab%20stars&query=%24.star_count&url=https%3A%2F%2Fgitlab.com%2Fapi%2Fv4%2Fprojects%2F31672379
    :target: https://img.shields.io/badge/dynamic/json?color=green&label=gitlab%20stars&query=%24.star_count&url=https%3A%2F%2Fgitlab.com%2Fapi%2Fv4%2Fprojects%2F31672379`
    :alt: Stars

.. image:: https://img.shields.io/pypi/pyversions/pyesmda.svg
    :target: https://pypi.org/pypi/pyesmda
    :alt: Python

.. image:: https://img.shields.io/pypi/v/pyesmda.svg
    :target: https://pypi.org/pypi/pyesmda
    :alt: PyPI

.. image:: https://pepy.tech/badge/pyesmda
    :target: https://pepy.tech/project/pyesmda
    :alt: Downoads

.. image:: https://gitlab.com/antoinecollet5/pyesmda/badges/master/pipeline.svg
    :target: https://gitlab.com/antoinecollet5/pyesmda/pipelines/
    :alt: Build Status

.. image:: https://readthedocs.org/projects/pyesmda/badge/?version=latest
    :target: https://pyesmda.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://gitlab.com/antoinecollet5/pyesmda/badges/master/coverage.svg
    :target: https://gitlab.com/antoinecollet5/pyesmda/pipelines/
    :alt: Coverage

.. image:: https://app.codacy.com/project/badge/Grade/bc4d1a8a1f574273a053a32d44931c00    
    :target: https://www.codacy.com/gl/antoinecollet5/pyesmda/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=antoinecollet5/pyesmda&amp;utm_campaign=Badge_Grade
    :alt: codacy
    
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
    :target: https://github.com/psf/black
    :alt: Black

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat
    :target: https://timothycrosley.github.io/isort
    :alt: isort

Python Ensemble Smoother with Multiple Data Assimilations

**pyesmda** is an open-source, pure python, and object-oriented library that provides 
a user friendly implementation of one of the most popular ensemble based method
for parameters estimation and data assimilation: the Ensemble Smoother with
Multiple Data Assimilations (ES-MDA) algorithm, introduced by Emerick and Reynolds [1-2].

Thanks to its simple formulation, ES-MDA of Emerick and Reynolds (2012) is perhaps the 
most used iterative form of the ensemble smoother in geoscience applications.

* Free software: MIT license
* Documentation: https://pyesmda.readthedocs.io.

The original python implementation was by Muhammad Iffan Hannanu
(https://github.com/iffanh/Playground).



References
----------

* [1] Emerick, A. A. and A. C. Reynolds, Ensemble smoother with multiple
  data assimilation, Computers & Geosciences, 2012.
* [2] Emerick, A. A. and A. C. Reynolds. (2013). History-Matching
  Production and Seismic Data in a Real Field Case Using the Ensemble
  Smoother With Multiple Data Assimilation. Society of Petroleum
  Engineers - SPE Reservoir Simulation Symposium
  1.    2. 10.2118/163675-MS.
