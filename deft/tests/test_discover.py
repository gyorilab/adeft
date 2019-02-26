from deft.discover import DeftMiner


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


def test_add():
    """Test the addition of candidates to the trie

    First add one maximal candidate. All nested parent candidates will be
    added as well. Check that the candidates are contained in the trie and
    that likelihood calculations are correct. Then add the parent of the
    original maximal candidate and check that likelihood has been updated
    correctly.
    """
    dm = DeftMiner('INDRA')
    candidate = ['the', 'integrated', 'network', 'and',
                 'dynamical', 'reasoning', 'assembler']
    dm._add(candidate)
    stemmed = ['assembl', 'reason', 'dynam', 'and',
               'network', 'integr', 'the']
    counts = [1]*7
    penalty = [1]*6 + [0]
    current = dm._internal_trie
    for penalty, token in zip(penalty, stemmed):
        assert token in current.children
        score = 1 - penalty
        assert current.children[token].score == score
        current = current.children[token]
    dm._add(candidate[1:])
    counts = [2]*6 + [1]
    penalty = [2]*5 + [1, 0]
    current = dm._internal_trie
    for count, penalty, token in zip(counts, penalty, stemmed):
        assert token in current.children
        score = count - penalty
        assert current.children[token].score == score
        current = current.children[token]


def test_process_texts():
    """Test processing of texts
    """
    dm = DeftMiner('INDRA')
    dm.process_texts([example_text1, example_text2,
                      example_text3, example_text4])
    assert dm.top()[0] == ('indonesian debt restructuring agency', 1.0)
    assert dm.top()[3] == ('integrated network and dynamical'
                           ' reasoning assembler', 1.0)
    assert dm.top()[7] == ('reasoning assembler', 0.0)

    # check that top works with limit
    assert dm.top(limit=5) == dm.top()[0:5]


def test_process_with_exclude():
    """Test processing of texts with excluded words"""
    dm = DeftMiner('INDRA', exclude='and')
    dm.process_texts([example_text1, example_text2,
                      example_text3, example_text4])
    assert dm.top()[0] == ('dynamical reasoning assembler', 2.0)
    assert dm.top()[1] == ('indonesian debt restructuring agency', 1.0)


def test_get_longforms():
    """Test breadth first search algorithm to extract longforms
    """
    dm = DeftMiner('INDRA')
    # ensure list of longforms is initialized correctly
    assert dm.top() == []
    
    dm.process_texts([example_text1, example_text2,
                      example_text3, example_text4])
    longforms = dm.get_longforms(cutoff=0.5)
    assert(len(longforms) == 2)
    assert longforms[0] == ('indonesian debt restructuring agency', 1.0)
    assert longforms[1] == ('integrated network and dynamical'
                            ' reasoning assembler', 1.0)
