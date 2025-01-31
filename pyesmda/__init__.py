"""
Purpose
=======

**pyesmda** is an open-source, pure python, and object-oriented library that provides
a user friendly implementation of one of the most popular ensemble based method
for parameters estimation and data assimilation: the Ensemble Smoother with
Multiple Data Assimilation (ES-MDA) algorithm, introduced by Emerick and Reynolds [1-2].

The following functionalities are directly provided on module-level.

Classes
=======

.. autosummary::
   :toctree: _autosummary

   ESMDA
   ESMDA_RS

Objective functions
===================

.. autosummary::
   :toctree: _autosummary

    compute_ensemble_average_normalized_objective_function
    compute_normalized_objective_function

Covariance approximation
========================

.. autosummary::
   :toctree: _autosummary

    get_ensemble_variance
    approximate_cov_mm
    approximate_covariance_matrix_from_ensembles
    inflate_ensemble_around_its_mean

Correlation functions
=====================

.. autosummary::
   :toctree: _autosummary

    distances_to_weights_beta_cumulative
    distances_to_weights_fifth_order

Other functions
===============

.. autosummary::
   :toctree: _autosummary

    check_nans_in_predictions

"""
from .__about__ import __author__, __version__
from .esmda import ESMDA
from .esmda_rs import ESMDA_RS
from .localization import (
    distances_to_weights_beta_cumulative,
    distances_to_weights_fifth_order,
)
from .utils import (
    approximate_cov_mm,
    approximate_covariance_matrix_from_ensembles,
    check_nans_in_predictions,
    compute_ensemble_average_normalized_objective_function,
    compute_normalized_objective_function,
    get_ensemble_variance,
    inflate_ensemble_around_its_mean,
)

__all__ = [
    "__version__",
    "__author__",
    "ESMDA",
    "ESMDA_RS",
    "get_ensemble_variance",
    "approximate_covariance_matrix_from_ensembles",
    "approximate_cov_mm",
    "compute_normalized_objective_function",
    "compute_ensemble_average_normalized_objective_function",
    "inflate_ensemble_around_its_mean",
    "check_nans_in_predictions",
    "distances_to_weights_beta_cumulative",
    "distances_to_weights_fifth_order",
]
