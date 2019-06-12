---
title: 'Adeft: Acromine-based Disambiguation of Entities from Text with applications to the biomedical literature
tags:
  - Python
  - natural language processing
  - text mining
  - named entity disambiguation
  - acronyms
  - Acromine
authors:
  - name: Albert Steppi
    orcid: 0000-0001-5871-6245
    affiliation: 1
  - name: Benjamin M. Gyori
    orcid: 0000-0001-9439-5346
    affiliation: 1
  - name: John A. Bachman
    orcid: 0000-0001-6095-2466
    affiliation: 1
affiliations:
 - name: Laboratory of Systems Pharmacology, Harvard Medical School
   index: 1
date: 12 June 2019
bibliography: paper.bib
---

# Summary

For machines to extract useful information from scientific documents, they must
be able to identify entities referenced in the text. For example, in the phrase
"binding of ligand to the IR is reduced", "IR" refers to the insulin receptor,
a gene with official symbol ``INSR``. This process of identification, known as
*named entity disambiguation,* requires the text string for the entity to be
mapped to an identifier in a database or ontology. A complicating factor is
that multiple distinct entities may be associated with the same text, leading
to ambiguity. In scientific and technical documents, this ambiguity frequently
originates from the use of overlapping acronyms or abbreviations: for example,
in the biomedical literature, the term "IR" can refer not only to the insulin
receptor, but also to ionizing radiation, ischemia reperfusion, insulin
resistance, and other concepts. While interpreting these ambiguities is rarely
a problem for human readers given the context of the whole document, it remains
a challenging problem for text mining tools, many of which process text
one sentence at a time.

``Adeft`` (Acromine-based Disambiguation of Entities From Text) is a Python
package for training and using statistical models to disambiguate named
entities in text based on document context. It is based on Acromine, a
previously-published algorithm that assembles a training corpus for the
different senses of an acronym by searching the text for defining patterns
(DPs) [@acromine1; @acromine2]. Defining patterns typically take the form of
parenthetical expressions, e.g. ``long form (shortform)``, which can be
identified systematically with regular expressions (for example, in the
preceding sentence, ``defining patterns (DPs)`` is a defining pattern).

Given a named entity shortform (e.g., "IR") and a set of texts containing the
shortform, Adeft first uses the Acromine algorithm to identify candidate
longforms (e.g., ``insulin receptor``, ``ionizing radiation``, etc.) by
searching for defining patterns. Second, the user selects the subset of
longforms relevant to their text mining use case and maps them to uniform
identifiers either manually or programmatically (e.g., "insulin receptor" is
mapped to gene symbol
[INSR](https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:6091),
whereas "ionizing radiation" is mapped to [MESH ID
D011839](https://www.ncbi.nlm.nih.gov/mesh?term=Radiation,%20Ionizing)). In
addition to its Python API, Adeft provides a simple web-based interface to
facilitate the  curation of these mappings. Third, Adeft stratifies the source
documents according to the defining patterns they contain, resulting in a
training corpus with multiple subsets of documents, one for each selected
longform.

Based on this training corpus, Adeft builds statistical models that can be used
to disambiguate an entity shortform given the full text of the document
containing the shortform. Adeft uses the Python package NLTK [@nltk] to
normalize the word frequencies for the documents in the training corpus by term
frequency-inverse document frequency (TF-IDF), then uses Scikit-learn
[@sklearn] to train logistic regression models to predict the entity identity
from the normalized word frequency vectors. Once trained, these models can be
used to disambiguate entities in new documents (including those not containing
the defining pattern).

In addition to the tools provided to build disambiguation models, Adeft also
facilitates the use of pre-trained models for **XXX** ambiguous acronyms
from the biomedical literature. However, the methods used by Adeft are not
specific to any particular domain or type of document. In addition to
code documentation, the Adeft repository contains Jupyter notebooks demonstrating Adeft workflows, including the use of pre-trained models and the training
and use of new ones.

# Acknowledgements

Development of this software was supported by the
Defense Advanced Research Projects Agency under award W911NF018-1-0124.

# References

