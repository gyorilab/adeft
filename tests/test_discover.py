import os
import uuid

from adeft.locations import TEST_RESOURCES_PATH
from adeft.discover import AdeftMiner, load_adeft_miner_from_dict, \
    load_adeft_miner, compose


# Path to scratch directory to write files to during tests
SCRATCH_PATH = os.path.join(TEST_RESOURCES_PATH, 'scratch')


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
    miner = AdeftMiner('INDRA')
    candidate = ['the', 'integrated', 'network', 'and',
                 'dynamical', 'reasoning', 'assembler']
    miner._add(candidate)
    stemmed = ['assembl', 'reason', 'dynam', 'and',
               'network', 'integr', 'the']
    counts = [1]*7
    penalty = [1]*6 + [0]
    current = miner._internal_trie
    for penalty, token in zip(penalty, stemmed):
        assert token in current.children
        score = 1 - penalty
        assert current.children[token].score == score
        current = current.children[token]
    miner._add(candidate[1:])
    counts = [2]*6 + [1]
    penalty = [2]*5 + [1, 0]
    current = miner._internal_trie
    for count, penalty, token in zip(counts, penalty, stemmed):
        assert token in current.children
        score = count - penalty
        assert current.children[token].score == score
        current = current.children[token]


def test_process_texts():
    """Test processing of texts
    """
    miner = AdeftMiner('INDRA')
    miner.process_texts([example_text1, example_text2,
                         example_text3, example_text4])
    top = miner.top()
    assert top[0] == ('indonesian debt restructuring agency', 2, 1.0)
    assert top[1][0] == ('integrated network and dynamical'
                         ' reasoning assembler')
    assert top[7] == ('reasoning assembler', 2, 0.0)

    # check that top works with limit
    assert miner.top(limit=5) == miner.top()[0:5]


def test_get_longforms():
    """Test breadth first search algorithm to extract longforms
    """
    miner = AdeftMiner('INDRA')
    # ensure list of longforms is initialized correctly
    assert miner.top() == []

    miner.process_texts([example_text1, example_text2,
                         example_text3, example_text4])
    longforms = miner.get_longforms()
    assert(len(longforms) == 2)
    assert longforms[0][0] == 'indonesian debt restructuring agency'
    assert longforms[0][1] >= 0.8
    assert longforms[1][0] == ('integrated network and dynamical'
                               ' reasoning assembler')
    assert longforms[1][1] >= 0.8


def test_miner_to_dict():
    miner = AdeftMiner('INDRA')
    miner.process_texts([example_text1, example_text2,
                         example_text3, example_text4])
    miner_dict = miner.to_dict()
    miner2 = load_adeft_miner_from_dict(miner_dict)
    assert miner.top() == miner2.top()
    assert miner.get_longforms(use_alignment_based_scoring=False) == \
        miner2.get_longforms(use_alignment_based_scoring=False)
    miner.compute_alignment_scores()
    assert miner.get_longforms() == miner2.get_longforms()


def test_serialize_adeft_miner():
    miner = AdeftMiner('INDRA')
    miner.process_texts([example_text1, example_text2,
                         example_text3, example_text4])
    temp_filename = os.path.join(SCRATCH_PATH, uuid.uuid4().hex)
    with open(temp_filename, 'w') as f:
        miner.dump(f)
    with open(temp_filename) as f:
        miner2 = load_adeft_miner(f)
    assert miner.top() == miner2.top()
    assert miner.get_longforms() == miner2.get_longforms()


def test_compose_adeft_miners():
    miner1 = AdeftMiner('INDRA')
    miner2 = AdeftMiner('INDRA')
    miner3 = AdeftMiner('INDRA')

    miner1.process_texts([example_text1, example_text2])
    miner2.process_texts([example_text3, example_text4])
    miner3.process_texts([example_text1, example_text2,
                          example_text3, example_text4])
    combined = compose(miner1, miner2)
    print(combined)
    assert combined.top() == miner3.top()


def test_prune():
    miner = AdeftMiner('INDRA')
    miner.process_texts([example_text1, example_text2,
                         example_text3, example_text4])
    candidates = [candidate for candidate, _, _ in miner.top()]
    miner.prune(5)
    pruned_candidates = [candidate for candidate, _, _ in miner.top()]
    assert pruned_candidates == [candidate for candidate in candidates if
                                 len(candidate.split()) <= 5]
