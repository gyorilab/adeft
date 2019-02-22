# Deft

The Deft Disambiguator (DD) is a utility for disambiguating acronyms and
abbreviations for genes in biological texts. It is based on a reimplementation
of the [Acromine](http://www.chokkan.org/research/acromine/) system developed
by the [NaCTeM](http://www.nactem.ac.uk/index.php) at the University of Manchester.
It makes use of pattern matching and machine learning to disambiguate abbreviations
based on text context.

## Installation

Deft works with Python versions 3.5 and above. To install, point pip to the
source repository at

    $ pip install git+https://github.com/indralab/deft.git

Deft's pretrained machine learning models can then be downloaded with the command

    $ python -m deft.download

## Using Deft
A list of available models can be imported with `from deft import available_models`

Here's an example of running a disambiguator for ER on a list of texts

```python
from deft.disambiguate import load_disambiguator

er_dd = load_disambiguator('ER')

    ...

er_dd.disambiguate(texts)
```

Users may also build and train their own disambiguators. See the documention
for more info.


## Documentation

Documentation is available at
[https://deft.readthedocs.io](http://deft.readthedocs.io)
    

