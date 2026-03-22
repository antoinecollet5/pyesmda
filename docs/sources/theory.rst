.. _theory_ref:


=================
Theory
=================

What is the Ensemble Smoother with Multiple Data Assimilations (ESMDA) ?

If you are interested in the math behind :py:mod:`pyesmda`, you've comed to the right place. The following is an extract summary coming from the PhD of Antoine COLLET (see chapter 4.4.1, 4.4.3 and 4.4.4 in :cite:`colletAssistedHistoryMatching2024`).

General "black-box" framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although these three "black-box" or "derivative-free" inversion methods may at first appear to be unrelated, not least because their respective authors present them in this way and do not use the same notations, they are in fact very similar in that they minimize the same objective function and can be derived from the same approximation form of Newton's method: Gauss-Newton iterations. But they differ in the computation of certain matrices and in the implementation.

.. _sec_bayesian_framework:

Bayesian framework
------------------

Considering the measurement equation


.. math::

    \mathbf{d}_{\mathrm{obs}} = \mathcal{F}(\mathbf{s}) + \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_{\mathrm{obs}}),


where :math:`\mathcal{F}` designates the forward operator predictions masked in the data domain to match observations, the history matching problem consists in finding :math:`\mathbf{s}` knowing :math:`\mathbf{d}_{\mathrm{obs}}`. This is equivalent to maximizing the probability function :math:`P(\mathbf{s}|\mathbf{d}_{\mathrm{obs}})` to obtain :math:`\widehat{\mathbf{s}}` the maximum likelihood estimate (MLE). Using Bayes theorem, we can calculate the posterior probability density function (PDF) of an event that will occur based on another event that has already occurred with

.. math::

  \begin{aligned}
  \mathrm{Bayes' rule:} \quad P(\mathbf{s}|\mathbf{d}_{\mathrm{obs}}) &= P(\mathbf{s}) \times \dfrac{P(\mathbf{d}_{\mathrm{obs}}|\mathbf{s})}{P(\mathbf{d}_{\mathrm{obs}})},
  \\
  & \propto \underbrace{P(\mathbf{d}_{\mathrm{obs}}|\mathbf{s})}_{\mathrm{likelyhood}} \times \underbrace{P(\mathbf{s})}_{\mathrm{prior}},
  \end{aligned}


where :math:`P(\mathbf{d}_{\mathrm{obs}})` is neglected because it is always positive, it does not depend on :math:`\mathbf{s}` and it is very difficult to compute. Since the measurement errors are assumed to be Gaussian, i.e., :math:`\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{C}_{\mathrm{obs}})`, then :math:`P(\mathbf{d}_{\mathrm{obs}}|\mathbf{s})` is Gaussian, and noting that :math:`\mathbb{E}\left[\mathbf{d}_{\mathrm{obs}}|\mathbf{s}\right] = \mathcal{F}(\mathbf{s})` and :math:`\mathrm{cov}(\mathcal{F}(\mathbf{s})) = \mathbf{C}_{\mathrm{obs}}`, it follows that

.. math::

  P(\mathbf{d}_{\mathrm{obs}}|\mathbf{s}) = \dfrac{1}{(2\pi)^{N_{\mathrm{obs}}/2}} \dfrac{1}{\sqrt{\mathrm{det(\mathbf{C}_{\mathrm{obs}})}}} \exp \left( - \dfrac{1}{2}\left(\mathbf{d}_{\mathrm{obs}} - \mathcal{F}(\mathbf{s})\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1}\left(\mathbf{d}_{\mathrm{obs}} - \mathcal{F}(\mathbf{s})\right)  \right).


In addition, one always has some prior information about the plausibility of models. In the case of history matching (HM), this includes a geological prior model (hopefully stochastic) constructed from log, core and seismic data, as well as information about the depositional environment. A general assumption is that the updated parameter :math:`\mathbf{s}` is a realization of a random :math:`N_{\mathrm{s}}-\mathrm{dimensional}` vector :math:`\mathbf{S}`, which is (multivariate) Gaussian or multinormal with mean :math:`\mathbf{s}_{\mathrm{prior}}` and autocovariance :math:`\mathbf{C}_{\mathrm{prior}}`. In other words, :math:`\mathbf{S} \sim \mathcal{N}(\mathbf{s}_{\mathrm{prior}}, \mathbf{C}_{\mathrm{prior}})` and the probability density for :math:`\mathbf{S}` is

.. math::

  P(\mathbf{s}) = \dfrac{1}{(2\pi)^{N_{\mathrm{s}}/2}} \dfrac{1}{\sqrt{\mathrm{det(\mathbf{C}_{\mathrm{prior}})}}} \exp \left( - \dfrac{1}{2}\left(\mathbf{s} - \mathbf{s}_{\mathrm{prior}}\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{prior}}^{-1}\left(\mathbf{s} - \mathbf{s}_{\mathrm{prior}}\right)  \right).


Maximizing the product of the prior and the likelihood inserted into Bayes’ theorem yields

.. math::

  P(\mathbf{s}|\mathbf{d}_{\mathrm{obs}}) = a \exp\left( -\mathcal{J} \right),


with


.. math::
  :label: eq_J_MAP

  \begin{aligned}
  a &= \dfrac{1}{(2\pi)^{N_{\mathrm{obs}}/2}} \dfrac{1}{\sqrt{\mathrm{det(\mathbf{C}_{\mathrm{obs}})}}} \times \dfrac{1}{(2\pi)^{N_{\mathrm{s}}/2}} \dfrac{1}{\sqrt{\mathrm{det(\mathbf{C}_{\mathrm{prior}})}}},
  \\
  \mathcal{J}(\mathbf{s}) &=  \underbrace{\frac{1}{2} \left(\mathcal{F}(\mathbf{s}) - \mathbf{d}_{\mathrm{obs}}\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1}\left(\mathcal{F}(\mathbf{s}) - \mathbf{d}_{\mathrm{obs}}\right)}_{\mathrm{likelihood~(data misfit)}} \underbrace{+ \frac{1}{2} \left(\mathbf{s} - \mathbf{s}_{\mathrm{prior}}\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{prior}}^{-1}\left(\mathbf{s} - \mathbf{s}_{\mathrm{prior}}\right)}_{\mathrm{prior}},
  \end{aligned}


Maximizing :math:`P(\mathbf{s}|\mathbf{d}_{\mathrm{obs}})`, i.e., finding the MAP estimate (the mode of the posterior distribution), is the same as minimizing its negative logarithm, i.e., the negative log-likelihood :math:`\mathcal{J}`.

.. _gn_procedure:

Gauss-Newton iterations
-----------------------

The MAP or most likely values are obtained by minimizing :math:`\mathcal{J}` from :eq:`eq_J_MAP` with respect to the vector :math:`\mathbf{s}`. As explained earlier, based on the initial guess (or most recent "good solution") :math:`\mathbf{s}_{\ell}`, a Newton-type iterative approach leads to a new solution :math:`\mathbf{s}_{\ell+1}` according to

.. math::


  \mathbf{s}_{\ell+1} = \mathbf{s}_{\ell} - \gamma_{\ell} \left(\dfrac{\partial^{2} \mathcal{J} (\mathbf{s}_{\ell})}{\partial \mathbf{s}_{\ell}^{2}}\right)^{-1} \dfrac{\partial \mathcal{J} (\mathbf{s}_{\ell})}{\partial \mathbf{s}_{\ell}}.


Defining :math:`\mathbf{J}_{\ell}`, the :math:`(N_{\mathrm{obs}} \times N_{\mathrm{s}})` Jacobian matrix of :math:`\mathcal{F}` at :math:`\mathbf{s}_{\ell}` as

.. math::

  \mathbf{J}_{\ell} = \left.\dfrac{\partial \mathcal{F}}{\partial \mathbf{s}} \right|_{\mathbf{s}=\mathbf{s}_{\ell}} = \mathcal{F}(\mathbf{s}_{\ell})\nabla_{\mathrm{s}}^{\mathrm{T}} = \displaystyle \begin{bmatrix}
  \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{0}}{\partial s_{0}} & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{0}}{\partial s_{1}} & \ldots & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{0}}{\partial s_{N_{\mathrm{s}}-1}}  \\
  \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{1}}{\partial s_{0}} & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{1}}{\partial s_{1}} & \ldots & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{1}}{\partial s_{N_{\mathrm{s}}-1}} \\
  \vdots & \vdots & \ddots & \vdots \\
  \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{N_{\mathrm{obs}}-1}}{\partial s_{0}} & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{N_{\mathrm{obs}}-1}}{\partial s_{1}} & \ldots & \dfrac{\partial \mathcal{F}(\mathbf{s}_{\ell})_{N_{\mathrm{obs}}-1}}{\partial s_{N_{\mathrm{s}}-1}}
  \end{bmatrix},


with the gradient row vector operator :math:`\nabla_{\mathrm{s}}^{\mathrm{T}} = \begin{bmatrix} \dfrac{\partial.}{\partial s_{0}}, &  \ldots & \dfrac{\partial.}{\partial s_{N_{\mathrm{s}} - 1}} \end{bmatrix}` and the column vector :math:`\mathcal{F}(\mathbf{s}_{\ell})`, :math:`\mathcal{F}` is linearized around :math:`\mathbf{s}_{\ell}` with :math:`\mathcal{F}(\mathbf{s}_{\ell+1}) \approx \mathcal{F}(\mathbf{s}_{\ell}) + \mathbf{J}_{\ell}(\mathbf{\mathbf{s}_{\ell+1}} - \mathbf{s}_{\ell})`. Then, the gradient of :math:`\mathcal{J}` reads

.. math::
  :label: eq_grad_J_MAP

  \dfrac{\partial \mathcal{J} (\mathbf{s}_{\ell})}{\partial \mathbf{s}_{\ell}} = \mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \left(\mathcal{F}(\mathbf{s}_{\ell})- \mathbf{d}_{\mathrm{obs}}\right) + \mathbf{C}_{\mathrm{prior}}^{-1} \left(\mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}}\right).


The Hessian is approximated in the Gauss-Newton way, i.e., neglecting :math:`\nabla^{2}_{\mathrm{s}} \mathcal{F}(\mathbf{s}_{j})^{\mathrm{T}}`:

.. math::
  :label: eq_hess_J_MAP

  \begin{aligned}
  \dfrac{\partial^{2} \mathcal{J} (\mathbf{s}_{\ell})}{\partial (\mathbf{s}_{\ell})^{2}} & = \mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \nabla^{2}_{\mathbf{s}} \mathcal{F}(\mathbf{s}_{\ell})^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \left(\mathcal{F}(\mathbf{s}_{\ell}) - \mathbf{d}_{\mathrm{obs}}\right) + \mathbf{C}_{\mathrm{prior}}^{-1},
  \\
  & \approx \mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \mathbf{C}_{\mathrm{prior}}^{-1}.
  \end{aligned}



A Gauss-Newton iteration is then defined as

.. math::
  :label: eq_gauss_newton_no_lemma

  \begin{aligned}
  \mathbf{s}_{j, \ell+1} & = \mathbf{s}_{\ell} &- \gamma_{\ell} & \left(\dfrac{\partial^{2} \mathcal{J} (\mathbf{s}_{\ell})}{\partial \mathbf{s}_{\ell}^{2}}\right)^{-1} \dfrac{\partial \mathcal{J} (\mathbf{s}_{\ell})}{\partial \mathbf{s}_{\ell}},
  \\
  & = \mathbf{s}_{\ell} &- \gamma_{\ell} & \Bigg(\mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \mathbf{C}_{\mathrm{prior}}^{-1}\Bigg)^{-1} \Bigg(\mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \left(\mathcal{F}(\mathbf{s}_{\ell}) - \mathbf{d}_{\mathrm{obs}}\right) + \mathbf{C}_{\mathrm{prior}}^{-1} \left(\mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}}\right)\Bigg),
  \\
  & = \mathbf{s}_{\ell} &- \gamma_{\ell} & \Bigg(\mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \mathbf{C}_{\mathrm{prior}}^{-1}\Bigg)^{-1} \mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \left(\mathcal{F}(\mathbf{s}_{\ell}) -  \mathbf{d}_{\mathrm{obs}}\right),
  \\
  & &- \gamma_{\ell} & \Bigg(\mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \mathbf{C}_{\mathrm{prior}}^{-1}\Bigg)^{-1} \mathbf{C}_{\mathrm{prior}}^{-1} \left(\mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}}\right).
  \end{aligned}

Since inverting large covariance matrices is not practical, we use the two following matrix inversion lemmas :cite:p:`golubMatrixComputations1996,petersenMatrixCookbook2008` for the second and third right-hand side terms of :eq:`eq_gauss_newton_no_lemma` respectively

.. math::
  :label: eq_mat_lemmas

  \begin{aligned}
  & \displaystyle \left(\mathbf{A}^{-1}+\mathbf{U}\mathbf{C}^{-1}\mathbf{V}\right)^{-1} \mathbf{U} \mathbf{C}^{-1}=\mathbf{A}\mathbf{U}\left(\mathbf{C}+\mathbf{VAU}\right)^{-1},
  \\
  & \displaystyle \left(\mathbf{A}^{-1}+\mathbf{U}\mathbf{C}^{-1}\mathbf{V}\right)^{-1}=\mathbf{A}-\mathbf{A}\mathbf{U}\left(\mathbf{C}+\mathbf{VAU}\right)^{-1}\mathbf{VA},
  \end{aligned}



with :math:`\mathbf{A} = \mathbf{C}_{\mathrm{prior}}`, :math:`\mathbf{C} = \mathbf{C}_{\mathrm{obs}}`, :math:`\mathbf{U} = \mathbf{J}_{\ell}^{\mathrm{T}}` and :math:`\mathbf{V} = \mathbf{J}_{\ell}`, which gives

.. math::
  :label: eq_gauss_newton_update_base

  \begin{aligned}
  \mathbf{s}_{\ell+1} & = \mathbf{s}_{\ell} - \gamma_{\ell} \Bigg[&&\mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} \Bigg(\mathbf{J}_{\ell} \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} + \mathbf{C}_{\mathrm{obs}}\Bigg)^{-1} \Big(\mathcal{F}(\mathbf{s}_{\ell}) -  \mathbf{d}_{\mathrm{obs}}\Big)
  \\
  & && + \mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}} - \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} \Bigg(\mathbf{J}_{\ell} \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} + \mathbf{C}_{\mathrm{obs}}\Bigg)^{-1} \mathbf{J}_{\ell} \Big(\mathbf{s}_{j,l} - \mathbf{s}_{\mathrm{prior}}\Big) \Bigg]
  \\
  & = \mathbf{s}_{\ell} - \gamma_{\ell} \Bigg[ && \mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}} + \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} \Bigg(\mathbf{J}_{\ell} \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} + \mathbf{C}_{\mathrm{obs}}\Bigg)^{-1} \Big(\mathcal{F}(\mathbf{s}_{\ell}) -  \mathbf{d}_{\mathrm{obs}} - \mathbf{J}_{\ell} \Big(\mathbf{s}_{\ell} - \mathbf{s}_{\mathrm{prior}}\Big)\Big) \Bigg].
  \end{aligned}

As previously stated, all implementations described below are derived from this last equation but differ in 1) the way :math:`\mathbf{J}_{\ell}` is approximated, 2) the representation of :math:`\mathbf{C}_{\mathrm{prior}}`, 3) how matrix inversions are performed, and 4) how the step length :math:`\gamma` is chosen.

Uncertainty quantification
--------------------------

Minimizing :math:`\mathcal{J}` allows to find the MAP :math:`\widehat{\mathbf{s}}` i.e., the mean of the PDF but it does not answer the question of how to sample the full PDF (uncertainty quantification). A classic way relies on the linearization of the measurement operator (through the sensitivity matrix :math:`\mathbf{J}`) which, under the assumptions made in :ref:`sec_bayesian_framework`, yields a local Gaussian for the posterior PDF. This local Gaussian is specified by its mean (the MAP) and the posterior covariance matrix :math:`\mathbf{C}_{\mathrm{post}}` which can be approximated by the inverse of the Hessian of the negative log-likelihood of the posterior PDF computed at the MAP estimate :cite:p:`kitanidisQuasiLinearGeostatisticalTheory1995,lepineUncertaintyAnalysisPredictive1999,tarantolaInverseProblemTheory2005`. Considering iteration :math:`\ell` at which :math:`\mathbf{s}_{\ell} = \widehat{\mathbf{s}}`,

.. math::


  \mathbf{C}_{\mathrm{ss}, \ell} \approx \left(\mathbf{H}_{\ell}\right)^{-1} \approx \left(\mathbf{J}_{\ell}^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1} \mathbf{J}_{\ell} + \mathbf{C}_{\mathrm{prior}}^{-1}\right)^{-1}.


Using the second lemma in :eq:`eq_mat_lemmas`, it gives

.. math::
  :label: eq_approx_cov_post_gn

  \mathbf{C}_{\mathrm{ss},\ell} \approx \mathbf{C}_{\mathrm{prior}} - \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} \left(\mathbf{C}_{\mathrm{obs}} + \mathbf{J}_{\ell} \mathbf{C}_{\mathrm{prior}} \mathbf{J}_{\ell}^{\mathrm{T}} \right)^{-1} \mathbf{J}_{\ell} \mathbf{C}_{\mathrm{prior}},


and using the forward operator linearization, the posterior covariance matrix on predictions is expressed following :cite:t:`lepineUncertaintyAnalysisPredictive1999`

.. math::
  :label: eq_cdd_analytical

  \mathbf{C}_{\mathrm{dd},\ell} \approx \mathbf{J}_{\ell} \mathbf{C}_{\mathrm{ss},\ell} \mathbf{J}_{\ell}^{\mathbf{T}}.




For large-scale systems, computing and storing the approximation to :math:`\mathbf{C}_{\mathrm{ss}}` is computationally infeasible because the prior covariance matrices arise from finely discretized fields and certain covariance kernels are dense :cite:p:`saibabaFastComputationUncertainty2015`. In addition, computing the dense measurement operator requires solving many forward PDE problems, which can be computationally intractable. Note also that when a quasi-Newton optimization is used such as L-BFGS-B, the BFGS approximation may not converge to the true Hessian matrix :cite:p:`ren-puConvergenceVariableMetric1983`, hence, this approximation can not be used as a posterior covariance matrix. To ovecome these issues, uncertainty analysis and optimization can be conducted using randomized sampling.

.. _sec_rml:

Randomized Maximum Likelihood (RML)
-----------------------------------

Practically, a rigorous sampling procedure of the PDF, e.g., Markov chain Monte Carlo aka MCMC :cite:p:`bonet-cunhaHybridMarkovChain1996,oliverMarkovChainMonte1997` is untractable for large-scale problems and approximate sampling methods must be used :cite:p:`emerickInvestigationSamplingPerformance2013`. The Randomized Maximum Likelihood approach is one of them. It was introduced by both :cite:t:`kitanidisQuasiLinearGeostatisticalTheory1995` and :cite:t:`oliverConditioningPermeabilityFields1996` and consists of sampling both from the prior distribution :math:`\mathbf{s} \sim \mathcal{N}(\mathbf{s}_{\mathrm{prior}},\mathbf{C}_{\mathrm{prior}})` and the measurements distribution :math:`\mathbf{d}_{\mathrm{uc}} \sim \mathcal{N}(\mathbf{d}_{\mathrm{obs}},\mathbf{C}_{\mathrm{obs}})`, forming a set of :math:`N_{e}` couples of "perturbed" parameter and observation vectors {:math:`\mathbf{s}_{j}`, :math:`\mathbf{d}_{\mathrm{uc}, j}`}, also called realizations. The subscript "uc" stands for unconditional because :math:`\mathbf{d}_{\mathrm{uc}_{j}} = \mathbf{d}_{\mathrm{obs}} + \mathbf{v}_{j}`, with :math:`\mathbf{v}_{j}` being an unconditional realization of :math:`\mathbf{C}_{\mathrm{obs}}` with zero mean. Instead of finding a single vector :math:`\widehat{\mathbf{s}}` with :math:`\mathbf{s}_{0} = \mathbf{s}_{\mathrm{prior}}` as an initial guess, RML requires solving :math:`N_{e}` independent minimization problems -- one for each draw which are used as initial guess and measurement vectors -- with the following modified stochastic cost function

.. math::
  :label: eq_J_RML

  \mathcal{J}(\mathbf{s}_{j}) =  \underbrace{\frac{1}{2} \left(\mathcal{F}(\mathbf{s}_{j}) - \mathbf{d}_{\mathrm{uc},j}\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{obs}}^{-1}\left(\mathcal{F}(\mathbf{s}_{j}) - \mathbf{d}_{\mathrm{uc}, j}\right)}_{\mathrm{likelihood (data misfit)}} \underbrace{+ \frac{1}{2} \left(\mathbf{s}_{j} - \mathbf{s}_{\mathrm{prior}}\right)^{\mathrm{T}} \mathbf{C}_{\mathrm{prior}}^{-1}\left(\mathbf{s}_{j} - \mathbf{s}_{\mathrm{prior}}\right)}_{\mathrm{prior}},


where :math:`j` denotes the :math:`j^{\mathrm{th}}` draw. After optimizing the :math:`N_{e}` problems, one obtains two ensembles of posterior parameter and prediction vectors that can be written under matrix form as

.. math::

  \begin{aligned}
  \mathbf{S} &= \begin{pmatrix}\mathbf{s}_{0}, & \mathbf{s}_{1}, & \dots, & \mathbf{s}_{N_{e}-1}\end{pmatrix},
  \\
  \mathbf{D} &= \begin{pmatrix}\mathbf{d}_{0}, & \mathbf{d}_{1}, & \dots, & \mathbf{d}_{N_{e}-1}\end{pmatrix},
  \end{aligned}

with shape (:math:`N_{\mathrm{s}} \times N_{e}`) and (:math:`N_{\mathrm{obs}} \times N_{e}`) respectively. Given two samples consisting of :math:`N_{e}` independent realizations :math:`\mathbf{x}_{0}, ..., \mathbf{x}_{N_{e}-1}` and :math:`\mathbf{y}_{0}, \dots, \mathbf{y}_{N_{e}-1}` of :math:`N_{x}` and :math:`N_{y}-\mathrm{dimensional}` vectors :math:`\mathbf{X} \in \mathbb{R}^{N_{x}\times 1}` and :math:`\mathbf{Y} \in \mathbb{R}^{N_{y}\times 1}`, an unbiased estimator of the covariance matrix

.. math::


  \mathrm{cov}(\mathbf{X},\mathbf{Y}) = \mathbb{E}\left[\left(\mathbf{X}-\mathbb{E}[\mathbf{X}]\right)\left(\mathbf{Y}-\mathbb{E}[\mathbf{Y}]\right)^{\mathrm{T}}\right],



is the empirical (or sample) covariance matrix denoted by a :math:`\sim`

.. math::
  :label: eq_empirical_cross_covariance

  \widetilde{\mathbf{C}} = \frac{1}{N_{e} - 1} \sum_{j=0}^{N_{e}-1}\left(\mathbf{x}_{j} - \overline{\mathbf{x}}\right)\left(\mathbf{x}_{j} - \overline{\mathbf{x}} \right)^{\mathrm{T}}.




The empirical auto-covariance matrices for :math:`\mathbf{s}` and :math:`\mathbf{d}` can then be computed from the ensembles :math:`\mathbf{S}` and :math:`\mathbf{D}` as

.. math::
  :label: eq_empirical_cross_covariances

  \begin{aligned}
  \widetilde{\mathbf{C}}_{\mathrm{SS}} & = \frac{1}{N_{e} - 1} \sum_{j=0}^{N_{e}-1}\left(\mathbf{s}_{j} - \overline{\mathbf{s}}\right)\left(\mathbf{s}_{j} - \overline{\mathbf{s}} \right)^{\mathrm{T}}  = \mathbf{A}_\mathrm{s}\mathbf{A}_\mathrm{s}^{\mathrm{T}},
  \\
  \widetilde{\mathbf{C}}^{f}_{\mathrm{DD}} &= \frac{1}{N_{e} - 1} \sum_{j=0}^{N_{e}-1}\left(\mathbf{d}_{j} -\overline{\mathbf{d}} \right)\left(\mathbf{d}_{j} - \overline{\mathbf{d}} \right)^{\mathrm{T}} = \mathbf{A}_\mathrm{d}\mathbf{A}_\mathrm{d}^{\mathrm{T}},
  \\
  \end{aligned}

with :math:`\mathbf{A}_\mathrm{s}` and :math:`\mathbf{A}_\mathrm{d}` the centered anomaly matrices with size (:math:`N_{\mathrm{s}} \times N_{e}`) and (:math:`N_{\mathrm{obs}} \times N_{e}`) defined as

.. math::
  :label: eq_anomaly_matrices

  \begin{aligned}
  \mathbf{A}_{\mathrm{s}} &= \dfrac{1}{\sqrt{N_{e}-1}}\mathbf{S} \left(\mathbf{I}_{N_{e}} - \dfrac{1}{N_{e}} \mathbf{11}^{\mathrm{T}} \right),
  \\
  \mathbf{A}_{\mathrm{d}} &= \dfrac{1}{\sqrt{N_{e}-1}}\mathbf{D} \left(\mathbf{I}_{N_{e}} - \dfrac{1}{N_{e}} \mathbf{11}^{\mathrm{T}} \right).
  \end{aligned}


where :math:`\mathbf{1} \in \mathbb{R}^{N_e}` is defined as a column vector with all elements equal to 1, i.e., :math:`\mathbf{11}^{\mathrm{T}}` is a (:math:`N_e \times N_e`) matrix filled with one, :math:`\mathbf{I}_{N_e}` is the :math:`N_e\mathrm{-dimensional}` identity matrix, and the projection  :math:`\left(\mathbf{I}_{N_{e}} - \dfrac{1}{N_{e}} \mathbf{1}^{\mathrm{T}} \right)` subtracts the mean from the ensemble :cite:p:`evensenEfficientImplementationIterative2019`. Assuming that :math:`\overline{\mathbf{d}} = \mathcal{F}(\overline{\mathbf{s}})`, a first-order Taylor series expansion gives

.. math::
  :label: eq_1st_order_taylor_cov

  \mathbf{d}_{j} - \overline{\mathbf{d}} = \mathcal{F}\left(\mathbf{s}_{j}\right) - \mathcal{F}(\overline{\mathbf{s}}) = \mathbf{J} \left(\mathbf{s}_{j} - \overline{\mathbf{s}}\right).

Injecting the previous results in the auto-covariance definitions :eq:`eq_empirical_cross_covariances` yields

.. math::


  \begin{aligned}
  \widetilde{\mathbf{C}}_{\mathrm{DD}} &= \frac{1}{N_{e} - 1} \sum_{j=0}^{N_{e}-1}\left(\mathbf{d}_{j} -\overline{\mathbf{d}} \right)\left(\mathbf{d}_{j} - \overline{\mathbf{d}} \right)^{\mathrm{T}},
  \\
  & = \frac{1}{N_{e} - 1} \sum_{j=0}^{N_{e}-1} \mathbf{J} \left(\mathbf{s}_{j} - \overline{\mathbf{s}}\right) \left(\mathbf{s}_{j} - \overline{\mathbf{s}}\right)^{\mathrm{T}} \mathbf{J}^{\mathrm{T}} = \mathbf{J} \widetilde{\mathbf{C}}_{\mathrm{SS}} \mathbf{J}^{\mathrm{T}},
  \end{aligned}

which is consistent with the analytical expression of :math:`\mathbf{C}_{\mathrm{dd}}` given in :eq:`eq_cdd_analytical`. When used with Gauss-Newton or quasi-Newton optimization coupled to an adjoint state model, the method has been shown to provide correct sampling of the PDF, giving as good a data fit as that generated by MCMC, even for highly nonlinear models :cite:p:`emerickInvestigationSamplingPerformance2013`. However, the cost of the method is very high because each optimization problem is solved independently. We will see further that the ensemble-based methods used in this manuscript, ESMDA and SIES, rely on the same idea of randomized sampling but also use the ensemble to compute the sensitivity matrices, thereby avoiding the need for an adjoint state and drastically reducing the number of runs required, while maintaining a correct PDF sampling.
