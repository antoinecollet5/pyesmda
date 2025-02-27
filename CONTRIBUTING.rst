.. highlight:: shell

============
Contributing
============

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Reporting of Bugs and Defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A defect is any variance between actual and expected result, this can include bugs in the code or defects in the documentation or visualization.

Please report defects to the `the GitHub Tracker <https://github.com/antoinecollet5/pyesmda/issues>`_
using the **Defect** description template.

`Merge Request Guidelines`_ for details on best developmental practices.

Features
~~~~~~~~

If you wish to propose a feature, please file an issue on `the GitHub Tracker <https://github.com/antoinecollet5/pyesmda/issues>`_ using the **Feature** description template. Community members will help refine and design your idea until it is ready for implementation.
Via these early reviews, we hope to steer contributors away from producing work outside of the project boundaries.

Please see the `Merge Request Guidelines`_ for details on best developmental practices.

Documentation
~~~~~~~~~~~~~

pyESMDA could always use more documentation, whether as part of the official pyESMDA docs, in docstrings, tutorials and even on the web in blog posts, articles and such.

For docstrings, please use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Working on issues
-----------------

After an issue is created, the progress of the issues is tracked on the `GitHub issue board <https://github.com/antoinecollet5/pyesmda/issues>`_.
The maintainers will update the state using `labels <https://github.com/antoinecollet5/pyesmda/labels>`_ .
Once an issue is ready for review a Merge Request can be opened.

Code Style
~~~~~~~~~~

The code is formatted with `Ruff <https://docs.astral.sh/ruff/>`__ so you don't have to worry about code formatting.
Docstrings follow `NumPy's style guide <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__ and all public
functions, classes and methods must come with a full description of input parameters and return outputs.
To format both code and docstrings, simply run the command:

Commit Style
~~~~~~~~~~~~

- The project follows `google's trunk based development <https://cloud.google.com/solutions/devops/devops-tech-trunk-based-development>`_ .
- The features are developed in feature branches and merged in the develop branch. The develop branch is periodically merged in the master branch for official releases.
- The commits follow the following naming conventions:
    + ENH: Enhancements (features)
    + FIX: Bug fixes
    + IMP: Implementation without feature change (refactoring)
    + STYLE: Minor style changes (doc, typo)
    + TEST: Unit test changes

Branch name convention
~~~~~~~~~~~~~~~~~~~~~~

- Feature branches should be named : feature/my_feature_name
- Fix branches should be named : fix/my_fix_name
- Release branches should be named : release/vX.X.X

Merge Request Guidelines
------------------------

Please make merge requests into the *develop* branch (not the *master* branch). Each request should be self-contained and address a single issue on the tracker.

When committing a feature or fix branch into develop, the branch should be merged using squash option.

When committing between other branches (master / release) use a no fast-forward merge type.

Before you submit a merge request, check that it meets these guidelines:

1. New code should be fully tested; running pytest in coverage mode can help identify gaps.
2. Documentation is updated, this includes docstrings and any necessary changes to existing tutorials, user documentation and so forth. We use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
3. The CI pipelines should pass for all merge requests.

   - Check the status of the pipelines, the status is also reported in the merge request.
   - Pre-commit should pass.
   - No degradation in code coverage.
   - Documentation should build.
4. Ensure your merge request contains a clear description of the changes made and how it addresses the issue. If useful, add a screenshot to showcase your work to facilitate an easier review.

Congratulations! The maintainers will now review your work and suggest any necessary changes.
If no changes are required, a maintainer will "approve" the review.
If you are
Thank you very much
for your hard work in improving pyESMDA.


Setting up pyESMDA for local development
------------------------------------------------

Ready to contribute? Here's how to set up `pyESMDA` for local development.

1. Fork the `pyesmda` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@hublab.com:your_name_here/pyesmda.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv pyesmda
    $ cd pyesmda/
    $ pip install -e .
    $ pip install -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the pre-commit hooks, the tests and have test coverage::

    $ pre-commit run --all-files
    $ pytest --cov

  If you have worked on documentation instead of code you may want to preview how your docs look locally.
  You can build the docs locally using:

  .. code-block:: shell

      $ cd docs
      $ make html

  The docs will be located in `pyesmda/docs/build`.


6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a merge request through the GitHub website.


.. tip:: Autobuild documentation

    If you are working on documentation it can be useful to automatically rebuild
    the docs after every change. This can be done using the `sphinx-autobuild`
    package. Through the following command:


    .. code-block:: shell

        $ sphinx-autobuild docs docs/build/html

    The documentation will then be hosted on `localhost:8000`


    will look for tests located in the tests/measurement directory and report test coverage of the pyesmda/measurement module.
