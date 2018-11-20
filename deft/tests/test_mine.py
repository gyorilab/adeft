from nltk.tokenize import word_tokenize
from deft.mine import ContinuousMiner
from deft.mine import SnowCounter
from unittest import skip


example_text1 = ('The Integrated Network and Dynamical Reasoning Assembler'
                 ' (INDRA) is an automated model assembly system interfacing'
                 ' with NLP systems and databases to collect knowledge, and'
                 ' through a process of assembly, produce causal graphs and'
                 ' dynamical models. INDRA draws on natural language'
                 ' processing systems and structured databases to collect'
                 ' mechanistic and causal assertions, represents them in'
                 ' standardized form (INDRA statements), and assembles them'
                 ' into various modeling formalisms including causal graphs'
                 ' and dynamical models')

example_text2 = ('Integrated Network and Dynamical Reasoning Assembler'
                 ' (INDRA) generates executable models of pathway dynamics'
                 ' from natural language.')

example_text3 = ('The Indonesian Debt Restructuring Agency (INDRA) was'
                 ' established by the Jakarta Initiative in 1998.')

example_text4 = ('An Indonesian Debt Restructuring Agency (INDRA) was'
                 ' established to provide foreign-exchange cover for'
                 ' Indonesian corporations with foreign currency denominated'
                 ' debt.')


def test_snow_counter():
    """Test stemmer

    Stemmer should be able to keep track of the
    most frequent word mapped to a particular stem.
    """
    snow = SnowCounter()
    words = ['verb', 'verbs', 'verbing', 'verbed', 'noun', 'nouns', 'nouning',
             'nouned', 'verb', 'verb', 'verb', 'verbed', 'noun', 'nouns',
             'nouns', 'nouning', 'nouning']
    for word in words:
        snow.stem(word)
    assert set(snow.counts['verb'].items()) == set([('verb', 4),
                                                    ('verbed', 2),
                                                    ('verbing', 1),
                                                    ('verbs', 1)])
    assert set(snow.counts['noun'].items()) == set([('noun', 2),
                                                    ('nouned', 1),
                                                    ('nouning', 3),
                                                    ('nouns', 3)])
    assert snow.most_frequent('verb') == 'verb'
    assert snow.most_frequent('noun') == 'nouning'


@skip
def test_get_candidates():
    """Test extraction of maximal longform candidate from text
    """
    mine = ContinuousMiner('INDRA')
    tokens = word_tokenize(example_text1)
    assert mine._get_candidates(tokens) == ['the', 'integrated', 'network',
                                            'and', 'dynamical', 'reasoning',
                                            'assembler']


def test_add():
    """Test the addition of candidates to the trie

    First add one maximal candidate. All nested parent candidates will be
    added as well. Check that the candidates are contained in the trie and
    that likelihood calculations are correct. Then add the parent of the
    original maximal candidate and check that likelihood has been updated
    correctly.
    """
    mine = ContinuousMiner('INDRA')
    candidate = ['the', 'integrated', 'network', 'and',
                 'dynamical', 'reasoning', 'assembler']
    mine._add(candidate)
    stemmed = ['assembl', 'reason', 'dynam', 'and',
               'network', 'integr', 'the']
    counts = [1]*7
    penalty = [1]*6 + [0]
    current = mine._internal_trie
    for penalty, token in zip(penalty, stemmed):
        assert token in current.children
        score = 1 - penalty
        assert current.children[token].score == score
        current = current.children[token]
    mine._add(candidate[1:])
    counts = [2]*6 + [1]
    penalty = [2]*5 + [1, 0]
    current = mine._internal_trie
    for count, penalty, token in zip(counts, penalty, stemmed):
        assert token in current.children
        score = count - penalty
        assert current.children[token].score == score
        current = current.children[token]


def test_consume():
    """Test processing of corpuses
    """
    mine = ContinuousMiner('INDRA')
    mine.consume([example_text1, example_text2, example_text3, example_text4])
    assert mine.top()[0] == ('indonesian debt restructuring agency', 1.0)
    assert mine.top()[3] == ('integrated network and dynamical'
                             ' reasoning assembler', 1.0)
    assert mine.top()[7] == ('reasoning assembler', 0.0)


def test_get_longforms():
    """Test breadth first search algorithm to extract longforms
    """
    mine = ContinuousMiner('INDRA')
    mine.consume([example_text1, example_text2, example_text3, example_text4])
    longforms = mine.get_longforms(0.5)
    assert(len(longforms) == 2)
    assert longforms[0] == ('indonesian debt restructuring agency', 1.0)
    assert longforms[1] == ('integrated network and dynamical'
                            ' reasoning assembler', 1.0)
