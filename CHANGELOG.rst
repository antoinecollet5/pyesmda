==============
Changelog
==============

0.2.0 (2022-03-06)
------------------

* `!PR6 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/6>`_ The parameter `stdev_d` becomes `cov_d`.
* `!PR5 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/5>`_ The parameter `n_assimilation` becomes `n_assimilations`.
* `!PR4 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/4>`_ The parameter `stdev_m` is removed.
* `!PR3 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/3>`_ Type hints are now used in the library.
* `!PR2 <https://gitlab.com/antoinecollet5/pyesmda/-/merge_requests/2>`_ Add the possibility to save the history of m and d. This introduces a new knew
  keyword (boolean) for the constructor `save_ensembles_history`. 
  Note that the `m_mean` attribute is depreciated and two new attributes are 
  introduced: `m_history`, `d_history` respectively to access the successive
  parameter and predictions ensemble. 


0.1.0 (2021-11-28)
------------------

* First release on PyPI.
