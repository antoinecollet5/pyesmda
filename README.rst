=======
pyESMDA
=======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |ty| |DOI|

🐍 A python impementation of the famous Ensemble Smoother with Multiple Data Assimilation (ESMDA).

**pyesmda** is an open-source, and object-oriented library that provides
a user friendly implementation of one of the most popular ensemble based method
for parameters estimation and data assimilation: the Ensemble Smoother with
Multiple Data Assimilation (ES-MDA) algorithm, introduced by Emerick and Reynolds [1-2].

Thanks to its simple formulation, ES-MDA of Emerick and Reynolds (2012) is perhaps the
most used iterative form of the ensemble smoother in geoscience applications.

**The complete and up to date documentation can be found here**: https://pyesmda.readthedocs.io.

===============
🚀 Quick start
===============

To install `pyesmda`, the easiest way is through `pip`:

.. code-block::

    pip install pyesmda

Or alternatively using `conda`

.. code-block::

    conda install pyesmda

You might also clone the repository and install from source

.. code-block::

    pip install -e .

Once the installation is done, the `ESMDA` interface is ready to use. Let's illustrate
how to use the lib with is a simple example where amplitude and change factor parameters
of n exponential function are estimated:

.. code-block:: python

    import numpy as np
    from pyesmda import ESMDA, ESMDA_RS, ESMDA_DMC, ESMDAInversionType, FixedLocalization
    import logging

    # set logging level
    logging.basicConfig(level=logging.INFO)

    def exponential(p, x):
        """
        Simple exponential function with an amplitude and change factor.

        Parameters
        ----------
        p : tuple, list
            Parameters vector: amplitude i.e. initial value and change factor.
        x : np.array
            Independent variable (e.g. time).

        Returns
        -------
        np.array
            Result.

        """
        return p[0] * np.exp(x * p[1])


    def forward_model(m_ensemble, x):
        """
        Wrap the non-linear observation model (forward model).

        Function calling the non-linear observation model (forward model).
        for all ensemble members and returning the predicted data for
        each ensemble member.

        Parameters
        ----------
        m_ensemble : np.array
            Initial ensemble of N_{e} parameters vector..
        x : np.array
            Independent variable (e.g. time).

        Returns
        -------
        d_pred: np.array
            Predicted data for each ensemble member.
        """
        # Initiate an array of predicted results.
        d_pred = np.zeros([x.shape[0], m_ensemble.shape[1]])
        for j in range(m_ensemble.shape[1]):
            # Calling the forward model for each member of the ensemble
            d_pred[:, j] = exponential(m_ensemble[:, j], x)
        return d_pred

        # seed for the reproductibility
        seed = 0
        rng = np.random.default_rng(seed=seed)

        # a and b are the reference parameters that we look for
        a = 10.0
        b = -0.0020
        # timesteps
        x = np.arange(500)
        # Generate synthetic data used as observations
        obs = exponential((a, b), x) + rng.normal(0.0, 1.0, 500)

The optimal solution (a, b) can be found following:

.. code-block:: python

    # Initiate an ensemble of (a, b) parameters
    n_ensemble = 100  # size of the ensemble
    # Uniform law for the parameter a ensemble
    ma = rng.uniform(low=-10.0, high=50.0, size=n_ensemble)
    # ma = rng.normal(loc=20.0, scale=20.0, size=n_ensemble)
    # Uniform law for the parameter b ensemble
    mb = rng.uniform(low=-0.001, high=0.01, size=n_ensemble)
    # mb = rng.normal(loc=-.002, scale=.005, size=n_ensemble)
    # Prior ensemble
    m_ensemble = np.stack((ma, mb), axis=0)

    # Observation error covariance matrix
    cov_obs = np.ones(obs.size) * 2.0
    # cov_obs = np.diag([0.5] * obs.shape[0])
    # cov_obs = np.ones(obs.size)

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])

    # Number of assimilations
    n_assimilations = 4

    # Use a geometric suite (see procedure un evensen 2018) to compte alphas.
    # Also explained in Torrado 2021 (see her PhD manuscript.)
    cov_obs_inflation_geo = 1.2
    cov_obs_inflation_factors: list[float] = [1.1]
    for l in range(1, n_assimilations):
        cov_obs_inflation_factors.append(
            cov_obs_inflation_factors[l - 1] / cov_obs_inflation_geo
        )
    scaling_factor: float = np.sum(1 / np.array(cov_obs_inflation_factors))
    cov_obs_inflation_factors = [
        alpha * scaling_factor for alpha in cov_obs_inflation_factors
    ]

    np.testing.assert_almost_equal(sum(1.0 / np.array(cov_obs_inflation_factors)), 1.0)

    # This is just for the test
    cov_mm_inflation_factor = 1.2

    solver = ESMDA(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        n_assimilations=n_assimilations,
        cov_obs_inflation_factors=cov_obs_inflation_factors,
        cov_mm_inflation_factor=cov_mm_inflation_factor,
        m_bounds=m_bounds,
        save_ensembles_history=True,
        inversion_type=ESMDAInversionType.EXACT_CHOLESKY,
        seed=seed,
        truncation=0.99,
        logger=logging.getLogger("ESMDA"),
    )
    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2).all()

    # Get the approximated parameters
    a_approx, b_approx = np.average(solver.m_prior, axis=1)

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))


    solver.logger.info(f"a = {a_approx:.5f} +/- {a_std:.4E}")
    solver.logger.info(f"b = {b_approx:.5f} +/- {b_std: 4E}")

Which yields

.. code-block::

    INFO:ESMDA:Assimilation # 1
    INFO:ESMDA:Assimilation # 2
    INFO:ESMDA:Assimilation # 3
    INFO:ESMDA:Assimilation # 4
    INFO:ESMDA:a = 9.99314 +/- 1.3697E-01
    INFO:ESMDA:b = -0.00202 +/-  7.707785E-05

In the above example, the user has to define the number of assimilations as well as
the inflation factor(s). `pyesmda` implement two `automatic` variant of ESMDA,
namely the `restricted step` variant [3] through`ESMDA_RS` and the `data misfit controller`
variant [4] through `ESMDA_DMC`:

.. code-block:: python

    # Example with ESMDA_RS: no inflation factor provided nor number of assimilations

    # A priori estimated parameters covariance
    std_m_prior = np.array([30.0, 0.01])

    # This is just for the test
    cov_mm_inflation_factor: float = 0.9

    solver = ESMDA_RS(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        cov_mm_inflation_factor=cov_mm_inflation_factor,
        m_bounds=m_bounds,
        C_MD_localization=FixedLocalization(np.ones((m_ensemble.shape[0], obs.size))),
        C_DD_localization=FixedLocalization(np.ones((obs.size, obs.size))),
        save_ensembles_history=True,
        std_m_prior=std_m_prior,
        random_state=123,
        batch_size=1,
        is_parallel_analyse_step=True,
        logger=logging.getLogger("ESMDA-RS"),
    )
    # Call the ES-MDA-RS solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=1e-1).all()

    # Get the approximated parameters
    a_approx, b_approx = np.average(solver.m_prior, axis=1)

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))


    solver.logger.info(f"a = {a_approx:.5f} +/- {a_std:.4E}")
    solver.logger.info(f"b = {b_approx:.5f} +/- {b_std: 4E}")

Which yields:

.. code-block::

    INFO:ESMDA-RS:- Inflation factor = 4132.540
    INFO:ESMDA-RS:Assimilation # 2
    INFO:ESMDA-RS:- Inflation factor = 62.403
    INFO:ESMDA-RS:Assimilation # 3
    INFO:ESMDA-RS:- Inflation factor = 9.042
    INFO:ESMDA-RS:Assimilation # 4
    INFO:ESMDA-RS:- Inflation factor = 1.145
    INFO:ESMDA-RS:a = 9.81329 +/- 1.5023E-01
    INFO:ESMDA-RS:b = -0.00190 +/-  7.620613E-05

And here is how to use `ESMDA_DMC`:

.. code-block:: python

    # Note: no inflation factor provided nor number of assimilations
    # This is just for the test
    cov_mm_inflation_factor: float = 0.9

    solver = ESMDA_DMC(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        cov_mm_inflation_factor=cov_mm_inflation_factor,
        m_bounds=m_bounds,
        C_MD_localization=FixedLocalization(np.ones((m_ensemble.shape[0], obs.size))),
        C_DD_localization=FixedLocalization(np.ones((obs.size, obs.size))),
        save_ensembles_history=True,
        random_state=123,
        batch_size=1,
        is_parallel_analyse_step=True,
        logger=logging.getLogger("ESMDA-DMC"),
    )
    # Call the ES-MDA-RS solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2).all()

    # Get the approximated parameters
    a_approx, b_approx = np.average(solver.m_prior, axis=1)

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))


    solver.logger.info(f"a = {a_approx:.5f} +/- {a_std:.4E}")  # ty:ignore[possibly-missing-attribute]
    solver.logger.info(f"b = {b_approx:.5f} +/- {b_std: 4E}")  # ty:ignore[possibly-missing-attribute]

Which yields:

.. code-block::

    INFO:ESMDA-DMC:Assimilation # 2
    INFO:ESMDA-DMC:- Inflation factor = 661.601
    INFO:ESMDA-DMC:Assimilation # 3
    INFO:ESMDA-DMC:- Inflation factor = 181.950
    INFO:ESMDA-DMC:Assimilation # 4
    INFO:ESMDA-DMC:- Inflation factor = 12.253
    INFO:ESMDA-DMC:Assimilation # 5
    INFO:ESMDA-DMC:- Inflation factor = 1.097
    INFO:ESMDA-DMC:a = 9.78691 +/- 1.3310E-01
    INFO:ESMDA-DMC:b = -0.00191 +/-  6.493808E-05

See all use cases in the tutorials section of the `documentation <https://pyesmda.readthedocs.io/en/latest/usage.html>`_.
- TODO 2D example

===================
🛠️ Localization
===================

Here are some of the cool features that this implementation provides (to the best of our knowledge in 2025).

🏗️ Complete example with supporting paper coming Q1 2026.

- TODO:link to correlation matrices building
- TODO:link to example with localization

See all use cases in the tutorials section of the `documentation <https://pyesmda.readthedocs.io/en/latest/usage.html>`_.

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2023-2026, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/pyesmda/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoinecollet5@gmail.com
- GitHub: https://github.com/antoinecollet5/pyesmda

We welcome contributions!

===============
✨ How to Cite
===============

**Software/Code citation for pyESMDA:**

.. code-block::

    Antoine Collet. (2026). pyESMDA - Python Ensemble Smoother with Multiple Data Assimilation (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.7425670

=============
📚 References
=============

[1] Emerick, A. A. and A. C. Reynolds, Ensemble smoother with multiple data assimilation, Computers & Geosciences, 2012.

[2] Emerick, A. A. and A. C. Reynolds. (2013). History-Matching Production and Seismic Data in a Real Field Case Using the Ensemble Smoother With Multiple Data Assimilation. Society of Petroleum Engineers - SPE Reservoir Simulation Symposium 1.    2. 10.2118/163675-MS.

[3] Duc Le, Alexandre Emerick, and Albert Reynolds. An Adaptive Ensemble Smoother With Multiple Data Assimilation for Assisted History Matching. SPE Journal, June 2016. doi:10.2118/173214-PA.

[4] Marco Iglesias and Yuchen Yang. Adaptive regularisation for ensemble Kalman inversion. Inverse Problems, 37(2):025008, January 2021. doi:10.1088/1361-6420/abd29b.

* Free software: SPDX-License-Identifier: BSD-3-Clause

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/pyesmda/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/pyesmda.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/pyesmda/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/pyesmda.svg
    :target: https://pypi.org/pypi/pyesmda
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/pyesmda.svg
    :target: https://pypi.org/pypi/pyesmda
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/pyesmda
    :target: https://pepy.tech/project/pyesmda
    :alt: Downoads

.. |Build Status| image:: https://github.com/antoinecollet5/pyesmda/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/pyesmda/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/pyesmda/badge/?version=latest
    :target: https://pyesmda.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/pyesmda/branch/master/graph/badge.svg?token=ISE874MMOF
    :target: https://codecov.io/gh/antoinecollet5/pyesmda
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/d581e8505fbb470a8e9ea08475e393ae
    :target: https://app.codacy.com/gh/antoinecollet5/pyesmda/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7425670.svg
   :target: https://doi.org/10.5281/zenodo.7425670
