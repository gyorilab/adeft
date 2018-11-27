from deft.recognize import Recognizer

longforms_map = {('reticulum', 'endoplasm'): 'endoplasmic reticulum',
                 ('receptor', 'estrogen'): 'estrogen receptor',
                 ('alpha', 'receptor', 'estrogen'): 'estrogen receptor alpha',
                 ('reticular', 'endoplasm'): 'endoplasmic reticular',
                 ('room', 'emerg'): 'emergency room'}


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
    recognizer = Recognizer('ER', longforms_map)
    trie = recognizer._trie
    for key, longform in longforms_map.items():
        current = trie
        for index, token in enumerate(key):
            assert token in current.children
            if index < len(key) - 1:
                assert current.children[token].longform is None
            else:
                assert current.children[token].longform == longform
            current = current.children[token]


def test_search():
    """Test that searching for a longform in the trie works correctly"""
    recognizer = Recognizer('ER', longforms_map)
    example = (('room', 'emerg', 'non', 'of', 'type', 'some',
                'reduc', 'program', 'hmo', 'mandatori', ',', 'women', 'for'),
               'emergency room')
    assert recognizer._search(example[0]) == example[1]


def test_recognizer():
    """Test the recognizer end to end"""
    recognizer = Recognizer('ER', longforms_map)
    for text, longform in [example1, example2, example3, example4, example5]:
        if recognizer.recognize(text) != longform:
            print(text, longform)
        assert recognizer.recognize(text) == longform
