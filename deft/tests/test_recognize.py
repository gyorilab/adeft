from nltk.stem.snowball import EnglishStemmer

from deft.nlp import word_tokenize
from deft.recognize import DeftRecognizer

_stemmer = EnglishStemmer()

grounding_map = {'endoplasmic reticulum': 'MESH:D004721',
                 'estrogen receptor': 'HGNC:3467',
                 'estrogen receptor alpha': 'HGNC:3467',
                 'endoplasmic reticular': 'MESH:D004721',
                 'emergency room': 'ungrounded'}


example1 = ('The growth of estrogen receptor (ER)-positive breast cancer'
            ' is inhibited by all-trans-retinoinc acid (RA).',
            'HGNC:3467')

example2 = ('In Xenopus oocytes, overexpressed, unassembled alpha- and'
            ' beta-subunits of Xenopus Na, K-ATPase are retained in the'
            ' endoplasmic reticulum (ER) and are degraded with different'
            ' kinetics.',
            'MESH:D004721')

example3 = ('For women, mandatory HMO programs reduce some types of'
            ' non emergency room (ER) use, and increase reported unmet'
            ' need for medical care.',
            'ungrounded')

example4 = ('We have analyzed interaction of coactivators with the'
            ' wild-type estrogen receptor alpha (ER), HEG0, and a mutant,'
            'L546P-HEG0, which is constitutively active in several'
            ' transiently transfected cells and a HeLa line that stably'
            ' propagates an estrogen-sensitive reporter gene.',
            'HGNC:3467')

example5 = ('A number of studies showed that chemotherapeutic benefits'
            ' may result from targeting the endoplasmic reticular (ER)'
            ' stress signaling pathway',
            'MESH:D004721')


def test_init():
    """Test that the recognizers internal trie is initialized correctly"""
    dr = DeftRecognizer('ER', grounding_map)
    trie = dr._trie
    for longform, grounding in grounding_map.items():
        edges = tuple(_stemmer.stem(token)
                      for token in word_tokenize(longform))[::-1]
        current = trie
        for index, token in enumerate(edges):
            assert token in current.children
            if index < len(edges) - 1:
                assert current.children[token].grounding is None
            else:
                assert current.children[token].grounding == grounding
            current = current.children[token]


def test_search():
    """Test that searching for a longform in the trie works correctly"""
    dr = DeftRecognizer('ER', grounding_map)
    example = (('room', 'emerg', 'non', 'of', 'type', 'some',
                'reduc', 'program', 'hmo', 'mandatori', ',', 'women', 'for'),
               'ungrounded')
    assert dr._search(example[0]) == example[1]


def test_recognizer():
    """Test the recognizer end to end"""
    dr = DeftRecognizer('ER', grounding_map)
    for text, result in [example1, example2, example3, example4, example5]:
        longform = dr.recognize(text)
        assert longform.pop() == result
