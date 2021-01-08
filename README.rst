==============
Active Learner
==============

Active Learner for Phrase Extraction

Example
============
An Example project using Active Learner can be found at: https://github.com/huangbeidan/AL-test2

Highlights
============
- Support interactive learning for phrase labeling
- Easy to extend the existing methods
- Main steps can be found in lsh_analyzer.py

Installation
============
Enter virtual environment first:

.. code-block:: bash

    $ source venv/bin/activate

``active_learner`` depends on the following libraries:

      - dill
      - matplotlib
      - mmh3
      - numpy
      - py-entitymatching
      - PyQt5
      - snapy
      - tqdm

To install:

.. code-block:: bash

    $ pip install active-learner==1.1.0


Environment Setup
==========
- create input/ foler and put four required files under input/, which should come from Autophrase immediate results
- create empty output/ folder
- in venv/lib/python3.8/site-packages/py_entitymatching/__init__.py:

  - add these two line:

  .. code-block:: python

  >>> from py_entitymatching.labeler.labeler import _init_label_table
  >>> from py_entitymatching.labeler.labeler import _post_process_labelled_table

- in venv/lib/python3.8/site-packages/py_entitymatching/matcher/matcherutils.py

  - replace the following line with:

  .. code-block: python
  >>> # from  sklearn.preprocessing import Imputer
  >>> from sklearn.impute import SimpleImputer

  >>> # imp = Imputer(missing_values=missing_val, strategy=strategy, axis=axis)
  >>> imp = SimpleImputer(missing_values=missing_val, strategy=strategy, axis=axis)

- in venv4/lib/python3.8/site-packages/py_entitymatching/gui/table_gui.py

  - replace the following line with:

  .. code-block: python

  >>>  # table.set_value(idxv[i], cols[j], val)
  >>>   table.at[idxv[i], cols[j]] = val



Quickstart
==========
To start interactive engine

.. code-block:: python

    >>> from active_learner.lsh_analyzer import LSHAnalyzer
    >>> analyzer = analyzer = LSHAnalyzer()

There is example code in main.py

parameters:

``num_queries``:
    The number of phrases for presenting users for labeling, default is 5
``threshold_nlargest``:
    Threshold for choosing the high variance terms, default is 0.1

