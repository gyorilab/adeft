# adeft

Adeft (Acromine based Disambiguation of Entities From Text context)
is a utility for building models to disambiguate acronyms and other abbreviations of biological terms in the scientific literature. It makes use of an implementation of the [Acromine](http://www.chokkan.org/research/acromine/) algorithm developed
by the [NaCTeM](http://www.nactem.ac.uk/index.php) at the University of Manchester
to identify possible longform expansions for shortforms in a text corpus.
It allows users to build disambiguation models to disambiguate shortforms based
on their text context. A growing number of pretrained disambiguation models are publically available to download through adeft.
## Installation

Adeft works with Python versions 3.5 and above. It is available on PyPi and can be installed with the command

    $ pip install adeft

Adeft's pretrained machine learning models can then be downloaded with the command

    $ python -m adeft.download

## Using adeft
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
    

