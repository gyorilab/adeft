from deft.recognize import Recognizer
from deft.nlp import word_tokenize
from nltk.stem.snowball import EnglishStemmer

_snow = EnglishStemmer()

longforms = ['endoplasmic reticulum',
             'estrogen receptor',
             'estrogen receptor alpha',
             'endoplasmic reticular',
             'emergency room']


example1 = ('The growth of estrogen receptor (ER)-positive breast cancer'
            ' is inhibited by all-trans-retinoinc acid (RA).',
            'estrogen receptor')

example2 = ('Assembly of alpha- and beta-subunits in the endoplasmic'
            ' reticulum is a prerequisite for the structural and'
            ' functional maturation of oligomeric P-type ATPases. In'
            ' Xenopus oocytes, overexpressed, unassembled alpha- and'
            ' beta-subunits of Xenopus Na, K-ATPase are retained in the'
            ' endoplasmic reticulum (ER) and are degraded with different'
            ' kinetics.',
            'endoplasmic reticulum')

example3 = ('For women, mandatory HMO programs reduce some types of'
            ' non emergency room (ER) use, and increase reported unmet'
            ' need for medical care.',
            'emergency room')

example4 = ('We have analyzed interaction of coactivators with the'
            ' wild-type estrogen receptor alpha (ER), HEG0, and a mutant,'
            'L546P-HEG0, which is constitutively active in several'
            ' transiently transfected cells and a HeLa line that stably'
            ' propagates an estrogen-sensitive reporter gene',
            'estrogen receptor alpha')

example5 = ('A number of studies showed that chemotherapeutic benefits'
            ' may result from targeting the endoplasmic reticular (ER)'
            ' stress signaling pathway',
            'endoplasmic reticular')


def test_init():
    """Test that the recognizers internal trie is initialized correctly"""
    recognizer = Recognizer('ER', longforms)
    trie = recognizer._trie
    for longform in longforms:
        edges = tuple(_snow.stem(token)
                      for token in word_tokenize(longform))[::-1]
        current = trie
        for index, token in enumerate(edges):
            assert token in current.children
            if index < len(edges) - 1:
                assert current.children[token].longform is None
            else:
                assert current.children[token].longform == longform
            current = current.children[token]


def test_search():
    """Test that searching for a longform in the trie works correctly"""
    recognizer = Recognizer('ER', longforms)
    example = (('room', 'emerg', 'non', 'of', 'type', 'some',
                'reduc', 'program', 'hmo', 'mandatori', ',', 'women', 'for'),
               'emergency room')
    assert recognizer._search(example[0]) == example[1]


def test_recognizer():
    """Test the recognizer end to end"""
    recognizer = Recognizer('ER', longforms)
    for text, result in [example1, example2, example3, example4, example5]:
        longform, filtered_text = recognizer.recognize(text)
        assert longform.pop() == result
        assert '(ER)' not in filtered_text
