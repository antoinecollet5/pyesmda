==============
Changelog
==============

0.4.1 (2023-07-31)
------------------

* `!PR33 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/33>`_ FIX: the covariance inflation is now applied before the forecast step.

0.4.0 (2023-06-28)
------------------

* `!PR30 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/30>`_ ENH: Allow to perform the analyse step by batches of gridblocks and avoid the
  construction on large (N, M) correlation matrices. Three new parameters have been introduced - **batch_size**: Number of parameters that are assimilated at once. This option is
  available to overcome memory limitations when the number of updated parameters is large. In that case, the size of the covariance matrices tends to explode
  and the update step must be performed by chunks of parameters.
  **is_parallel_analyse_step**: Whether to use parallel computing for the analyse step if the number of batch is above one. The default is True.
  **n_batches**: Number of batches required during the update step.
  It also introduces the support for sparse matrices for correlation matrices and observation covariance matrix.
* `!PR29 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/29>`_ ENH: Add a function to get the ensemble variance. Convinient to compute the uncertainty a posteriori without the full covariance matrix approximation.
* STYLE: Format with updated version of black.

0.3.3 (2022-12-12)
------------------

* `!PR27 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/27>`_ STYLE: Add a DOI number from zenodo and correct typos.

0.3.2 (2022-10-07)
------------------

* `!PR21 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/21>`_ FIX: design - some static methods should be moved to a utils.py file.

0.3.1 (2022-08-12)
------------------

* `!PR20 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/20>`_ Fix ESMDA-RS documentation and change the
  **cov_m_prior** input parameter to its diagonal **std_m_prior** to be consistent with the implementation and be less memory consuming.

0.3.0 (2022-08-12)
------------------

* `!PR15 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/15>`_ Implement ESMDA-RS (restricted step) which provides
  an automatic estimation of the inflation parameter and determines when to stop (number of assimilations) on the fly.
* `!PR14 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/14>`_ Add keyword **is_forecast_for_last_assimilation** to choose whether to
  compute the predictions for the ensemble obtained at the last assimilation step. The default is True.
* `!PR13 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/13>`_ Implementation: Faster analyse step by avoiding matrix inversion.
* `!PR12 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/12>`_ Add a seed parameter for the random
  number generation **seed** in the prediction perturbation step.
  To avoid confusion , **cov_d** has been renamed **cov_obs**.
* `!PR11 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/11>`_ Implement the covariance localization. Introduces the
  correlation matrices **dd_correlation_matrix** and **md_correlation_matrix**.
  To avoid confusion , **cov_d** has been renamed **cov_obs**.
* `!PR10 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/10>`_ Implement the parameters auto-covariance inflation.
  Add the estimation of the parameters auto-covariance matrix. The parameter **alpha** becomes **cov_obs_inflation_factors**.


0.2.0 (2022-07-23)
------------------

* `!PR6 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/6>`_ The parameter **stdev_d** becomes **cov_d**.
* `!PR5 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/5>`_ The parameter **n_assimilation** becomes **n_assimilations**.
* `!PR4 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/4>`_ The parameter **stdev_m** is removed.
* `!PR3 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/3>`_ Type hints are now used in the library.
* `!PR2 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/2>`_ Add the possibility to save the history of m and d. This introduces a new knew
  keyword (boolean) for the constructor **save_ensembles_history**.
  Note that the **m_mean** attribute is depreciated and two new attributes are
  introduced: **m_history**, **d_history** respectively to access the successive
  parameter and predictions ensemble.


0.1.0 (2021-11-28)
------------------


* First release on PyPI.
