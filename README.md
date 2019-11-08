# Adeft
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build](https://travis-ci.org/indralab/adeft.svg)](https://travis-ci.org/indralab/adeft)
[![Documentation](https://readthedocs.org/projects/adeft/badge/?version=latest)](https://adeft.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/adeft.svg)](https://badge.fury.io/py/adeft)
[![Python 3](https://img.shields.io/pypi/pyversions/adeft.svg)](https://www.python.org/downloads/release/python-357/)

Adeft (Acromine based Disambiguation of Entities From Text context)
is a utility for building models to disambiguate acronyms and other abbreviations of biological terms in the scientific literature. It makes use of an implementation of the [Acromine](http://www.chokkan.org/research/acromine/) algorithm developed
by the [NaCTeM](http://www.nactem.ac.uk/index.php) at the University of Manchester
to identify possible longform expansions for shortforms in a text corpus.
It allows users to build disambiguation models to disambiguate shortforms based
on their text context. A growing number of pretrained disambiguation models are publicly available to download through adeft.

## Installation

Adeft works with Python versions 3.5 and above. It is available on PyPi and can be installed with the command

    $ pip install adeft

Adeft's pretrained machine learning models can then be downloaded with the command

    $ python -m adeft.download

If you choose to install by cloning this repository

    $ git clone https://github.com/indralab/adeft.git

You should also run

    $ python setup.py build_ext --inplace

at the top level of your local repository in order to build the extension module
for alignment based longform detection and scoring.

## Using Adeft
A dictionary of available models can be imported with `from adeft import available_models`

The dictionary maps shortforms to model names. It's possible for multiple equivalent
shortforms to map to the same model.

Here's an example of running a disambiguator for ER on a list of texts

```python
from adeft.disambiguate import load_disambiguator

er_dd = load_disambiguator('ER')

    ...

er_dd.disambiguate(texts)
```

Users may also build and train their own disambiguators. See the documention
for more info.


## Documentation

Documentation is available at
[https://adeft.readthedocs.io](http://adeft.readthedocs.io)

Jupyter notebooks illustrating Adeft workflows are available under `notebooks`:
- [Introduction](notebooks/introduction.ipynb)
- [Model building](notebooks/model_building.ipynb)


## Testing

Adeft uses `nosetests` for unit testing, and is integrated with the Travis
continuous integration environment. To run tests locally, make sure
to install the test-specific requirements listed in setup.py as

```bash
pip install adeft[test]
```

and download all pre-trained models as shown above.
Then run `nosetests` in the top-level `adeft` folder.

## Funding

Development of this software was supported by the Defense Advanced Research
Projects Agency under award W911NF018-1-0124 and the National Cancer Institute
under award U54-CA225088.
