import math
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


def test_get_candiates():
    mine = ContinuousMiner('INDRA')
    tokens = word_tokenize(example_text1)
    assert mine._get_candidates(tokens) == ['the', 'integrated', 'network',
                                            'and', 'dynamical', 'reasoning',
                                            'assembler']


def test_add():
    mine = ContinuousMiner('INDRA')
    candidate = ['the', 'integrated', 'network', 'and',
                 'dynamical', 'reasoning', 'assembler']
    mine._add(candidate)
    stemmed = ['assembl', 'reason', 'dynam', 'and',
               'network', 'integr', 'the']
    current = mine._internal_trie
    for index, token in enumerate(stemmed):
        assert token in current.children
        penalty = 0 if index == len(stemmed) - 1 else 1
        assert current.children[token].LH == math.log2(index+2) - penalty
        current = current.children[token]


@skip
def test_consume():
    mine = ContinuousMiner('INDRA')
    mine.consume([example_text1, example_text2, example_text3, example_text4])
    assert mine.top()[0][0] == ('integrated network and dynamical'
                                ' reasoning assembler')
    assert mine.top()[0][1] == 2*math.log(6)
    assert mine.top()[1][1] == 1
